import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import nibabel as nib
import psutil
import torch

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def calculate_psnr(reconstructed, real):
    """Calculate PSNR between two images."""
    if reconstructed.max() <= 1.0:
        reconstructed = (reconstructed * 255).astype(np.uint8)
    if real.max() <= 1.0:
        real = (real * 255).astype(np.uint8)
    return psnr(reconstructed, real)

def load_mask(mask_path):
    """Load the 3D mask from the given path."""
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata()
    return mask

def apply_mask(image, mask):
    """Apply the mask to the image."""
    return image * mask

def calculate_masked_psnr(reconstructed, real, mask):
    """Calculate PSNR between two masked images."""
    reconstructed_masked = apply_mask(reconstructed, mask)
    real_masked = apply_mask(real, mask)
    return calculate_psnr(reconstructed_masked, real_masked)

def pad_mask(mask, target_shape):
    """Pad the mask to the target shape with zeros."""
    pad_y = (target_shape[0] - mask.shape[0]) // 2
    pad_x = (target_shape[1] - mask.shape[1]) // 2
    pad_width = ((pad_y, target_shape[0] - mask.shape[0] - pad_y),
                 (pad_x, target_shape[1] - mask.shape[1] - pad_x))
    padded_mask = np.pad(mask, pad_width, mode='constant', constant_values=0)
    return padded_mask

def save_subplot_image(input_img1, input_img2, input_img3, reconstructed, real, psnr_value, output_path):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(input_img1, cmap='gray')
    axes[0].set_title('Input Image 1')

    axes[1].imshow(input_img2, cmap='gray')
    axes[1].set_title('Input Image 2')

    axes[2].imshow(input_img3, cmap='gray')
    axes[2].set_title('Input Image 3')

    axes[3].imshow(reconstructed, cmap='gray')
    axes[3].set_title(f'Reconstructed Image\nPSNR: {psnr_value:.2f}')

    axes[4].imshow(real, cmap='gray')
    axes[4].set_title('Real Image')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def plot_psnr_statistics(patient_avg_psnr, output_path):
    patients = list(patient_avg_psnr.keys())
    psnr_values = list(patient_avg_psnr.values())

    sorted_psnr_values = sorted(psnr_values)
    median_psnr = np.median(sorted_psnr_values)
    best_psnr = sorted_psnr_values[-1]
    worst_psnr = sorted_psnr_values[0]

    plt.figure(figsize=(10, 6))
    plt.plot(patients, psnr_values, label='PSNR Values', marker='o', linestyle='-')
    plt.axhline(y=median_psnr, color='r', linestyle='--', label=f'Median PSNR: {median_psnr:.2f}')
    plt.axhline(y=best_psnr, color='g', linestyle='--', label=f'Best PSNR: {best_psnr:.2f}')
    plt.axhline(y=worst_psnr, color='b', linestyle='--', label=f'Worst PSNR: {worst_psnr:.2f}')

    plt.xlabel('Patients')
    plt.ylabel('PSNR')
    plt.title('PSNR Statistics for Each Patient')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"memory usage:{process.memory_info().rss / 1024 ** 2} MB")

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.phase = 'train'  # Ensure the phase is set to 'test'

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of testing images = %d' % dataset_size)

    # set to test mode
    opt.phase = 'test'

    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.netG = 'unet_256'

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # Initialize logger
    log_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, 'test_log.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ])
    logger = logging.getLogger()

    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name,
                               config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    if opt.eval:
        model.eval()

    # Directory where masks are stored
    mask_dir = '/media/e210-pc34/Adi Hard disk/rakshana/mask'

    psnr_values = []  # List to store PSNR values
    patient_psnr = defaultdict(list)  # Dictionary to store PSNR values for each patient
    patient_images = defaultdict(list)  # Dictionary to store images and PSNR for each patient

    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals_test_3()  # get image results
        img_path = model.get_image_paths()  # get image paths

        # Split the input tensor into three images (A1, A2, A3)
        A1, A2, A3 = torch.chunk(data['A'], 3, dim=1)

        # Extract reconstructed and real images
        reconstructed_img = visuals['fake_B'][0, 0].cpu().numpy()  # Assuming the key for reconstructed image is 'fake_B'
        real_img = visuals['real_B'][0, 0].cpu().numpy()  # Assuming the key for real image is 'real_B'

        # Normalize images if necessary
        reconstructed_img = (reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min()) if reconstructed_img.min() < 0 or reconstructed_img.max() > 1 else reconstructed_img
        real_img = (real_img - real_img.min()) / (real_img.max() - real_img.min()) if real_img.min() < 0 or real_img.max() > 1 else real_img

        # Print normalization check
        print(f'Reconstructed image min: {reconstructed_img.min()}, max: {reconstructed_img.max()}')
        print(f'Real image min: {real_img.min()}, max: {real_img.max()}')

        # Extract patient ID and slice index from image path (assuming the patient ID and slice index are part of the filename)
        filename = os.path.basename(img_path[0])
        filename_parts = filename.split('_')
        patient_id = filename_parts[1]
        slice_index = int(filename_parts[3].split('.')[0])  # Extract slice index from filename

        # Load the corresponding 3D mask for the patient
        mask_path = os.path.join(mask_dir, f'BraTS2021_{patient_id}_t1ce_bet_mask.nii.gz')
        mask_3d = load_mask(mask_path)

        # Get the 2D slice from the 3D mask
        mask_slice = mask_3d[:, :, slice_index]

        # Pad the mask to match the image size
        padded_mask_slice = pad_mask(mask_slice, (256, 256))

        # Calculate masked PSNR
        psnr_value = calculate_masked_psnr(reconstructed_img, real_img, padded_mask_slice)
        psnr_values.append((img_path[0], psnr_value))

        patient_psnr[patient_id].append(psnr_value)
        patient_images[patient_id].append((A1.cpu().numpy(), A2.cpu().numpy(), A3.cpu().numpy(), reconstructed_img, real_img, psnr_value, filename))

        logger.info('Image %d (%s): PSNR = %f', i, img_path[0], psnr_value)
        if i % 100 == 0:
            print_memory_usage()
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                    use_wandb=opt.use_wandb)

    patient_avg_psnr = {patient: np.mean(values) for patient, values in patient_psnr.items()}
    best_patient = max(patient_avg_psnr, key=patient_avg_psnr.get)
    worst_patient = min(patient_avg_psnr, key=patient_avg_psnr.get)

    for patient, avg_psnr in patient_avg_psnr.items():
        logger.info('Patient %s: Average PSNR = %f', patient, avg_psnr)

    logger.info('Best Patient: %s with Average PSNR = %f', best_patient, patient_avg_psnr[best_patient])
    logger.info('Worst Patient: %s with Average PSNR = %f', worst_patient, patient_avg_psnr[worst_patient])

    with open(os.path.join(web_dir, 'psnr_values.txt'), 'w') as f:
        for img_path, value in psnr_values:
            f.write(f'Image ({img_path}): PSNR = {value}\n')
        f.write('\nAverage PSNR for each patient:\n')
        for patient, avg_psnr in patient_avg_psnr.items():
            f.write(f'Patient {patient}: Average PSNR = {avg_psnr}\n')
        f.write(f'\nBest Patient: {best_patient} with Average PSNR = {patient_avg_psnr[best_patient]}\n')
        f.write(f'Worst Patient: {worst_patient} with Average PSNR = {patient_avg_psnr[worst_patient]}\n')

    for patient_id, patient_data in patient_images.items():
        patient_dir = os.path.join(web_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        for A1_img, A2_img, A3_img, reconstructed_img, real_img, psnr_value, original_filename in patient_data:
            save_subplot_image(A1_img[0, 0], A2_img[0, 0], A3_img[0, 0], reconstructed_img, real_img, psnr_value, os.path.join(patient_dir, original_filename))

    plot_psnr_statistics(patient_avg_psnr, os.path.join(web_dir, 'psnr_statistics.png'))

    webpage.save()  # save the HTML
