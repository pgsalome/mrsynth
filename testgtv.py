import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import numpy as np
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import nibabel as nib
import psutil
import torch
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def calculate_psnr(reconstructed, real):
    return peak_signal_noise_ratio(reconstructed, real)

def calculate_ssim(reconstructed, real):
    return structural_similarity(reconstructed, real, data_range=real.max() - real.min())

def calculate_mse(reconstructed, real):
    return mean_squared_error(reconstructed, real)

def calculate_mae(reconstructed, real):
    return np.mean(np.abs(reconstructed - real))

def pad_mask(mask, target_shape):
    """Pad the mask to the target shape with zeros."""
    pad_y = (target_shape[0] - mask.shape[0]) // 2
    pad_x = (target_shape[1] - mask.shape[1]) // 2
    pad_width = ((pad_y, target_shape[0] - mask.shape[0] - pad_y),
                 (pad_x, target_shape[1] - mask.shape[1] - pad_x))
    padded_mask = np.pad(mask, pad_width, mode='constant', constant_values=0)
    return padded_mask

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

def calculate_masked_ssim(reconstructed, real, mask):
    """Calculate SSIM between two masked images."""
    reconstructed_masked = apply_mask(reconstructed, mask)
    real_masked = apply_mask(real, mask)
    return calculate_ssim(reconstructed_masked, real_masked)

def calculate_error_map(reconstructed, real):
    """Calculate the absolute difference between two images."""
    return np.abs(reconstructed - real)

def plot_error_map(error_map, output_path):
    """Plot and save the error map."""
    plt.figure(figsize=(6, 6))
    plt.imshow(error_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Error Map')
    plt.savefig(output_path)
    plt.close()

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
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")

import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.phase = 'test'  # Ensure the phase is set to 'test'

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of testing images = %d' % dataset_size)

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

    # Directory where GTV masks are stored
    gtv_mask_dir = '/media/e210-pc34/gjsalome/binarytestmask'  # Adjust this to the path where your GTV masks are stored

    # Initialize storage for 3D volumes as dictionaries with slice indices
    patient_real_volumes = defaultdict(dict)
    patient_reconstructed_volumes = defaultdict(dict)
    patient_gtv_volumes = defaultdict(dict)

    psnr_values = []  # List to store PSNR values
    ssim_values = []  # List to store SSIM values
    mse_values = []   # List to store MSE values
    mae_values = []   # List to store MAE values
    error_map_mean_values = []  # List to store mean values of error maps
    error_map_std_values = []   # List to store standard deviation of error maps
    patient_psnr = defaultdict(list)  # Dictionary to store PSNR values for each patient
    patient_ssim = defaultdict(list)  # Dictionary to store SSIM values for each patient
    patient_error_map_means = defaultdict(list)  # Dictionary to store mean error map values for each patient
    patient_error_map_stds = defaultdict(list)  # Dictionary to store std dev of error map values for each patient
    patient_mse = defaultdict(list)  # Dictionary to store MSE values for each patient
    patient_mae = defaultdict(list)  # Dictionary to store MAE values for each patient

    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals_test()  # get image results
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

        # Prepare the directory for each patient's error maps
        patient_dir = os.path.join(web_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        # Load the corresponding GTV mask for the patient
        gtv_mask_path = os.path.join(gtv_mask_dir, f'{patient_id}_MR_GTV.nii')
        gtv_mask_3d = load_mask(gtv_mask_path)

        # Get the 2D slice from the GTV mask
        gtv_mask_slice = gtv_mask_3d[:, :, slice_index]

        # Pad the GTV mask to match the image size
        padded_gtv_mask_slice = pad_mask(gtv_mask_slice, (256, 256))

        # Calculate metrics using the GTV mask instead of the CT1 mask
        psnr_value = calculate_masked_psnr(reconstructed_img, real_img, padded_gtv_mask_slice)
        ssim_value = calculate_masked_ssim(reconstructed_img, real_img, padded_gtv_mask_slice)
        mse_value = calculate_mse(reconstructed_img, real_img)
        mae_value = calculate_mae(reconstructed_img, real_img)

        psnr_values.append((img_path[0], psnr_value))
        ssim_values.append((img_path[0], ssim_value))
        mse_values.append((img_path[0], mse_value))
        mae_values.append((img_path[0], mae_value))

        patient_psnr[patient_id].append(psnr_value)
        patient_ssim[patient_id].append(ssim_value)
        patient_mse[patient_id].append(mse_value)
        patient_mae[patient_id].append(mae_value)

        logger.info('Image %d (%s): PSNR = %f, SSIM = %f, MSE = %f, MAE = %f', i, img_path[0], psnr_value, ssim_value, mse_value, mae_value)

        # Store slices for 3D volume creation
        patient_real_volumes[patient_id][slice_index] = real_img
        patient_reconstructed_volumes[patient_id][slice_index] = reconstructed_img
        patient_gtv_volumes[patient_id][slice_index] = padded_gtv_mask_slice

        # Calculate and save error map
        error_map = calculate_error_map(reconstructed_img, real_img)
        error_map_mean = np.mean(error_map)
        error_map_std = np.std(error_map)

        error_map_mean_values.append((img_path[0], error_map_mean))
        error_map_std_values.append((img_path[0], error_map_std))

        patient_error_map_means[patient_id].append(error_map_mean)
        patient_error_map_stds[patient_id].append(error_map_std)

        logger.info('Error Map Metrics for Image %d (%s): Mean Error = %f, Std Error = %f', i, img_path[0], error_map_mean, error_map_std)

        # Save the error map for this slice
        error_map_output_path = os.path.join(patient_dir, f'{filename}_error_map.png')
        plot_error_map(error_map, error_map_output_path)

        if i % 100 == 0:
            print_memory_usage()

    # After processing all slices, create 3D NIfTI volumes and save them
    for patient_id in patient_real_volumes:
        # Sort slices by index
        sorted_real_slices = [patient_real_volumes[patient_id][i] for i in sorted(patient_real_volumes[patient_id].keys())]
        sorted_reconstructed_slices = [patient_reconstructed_volumes[patient_id][i] for i in sorted(patient_reconstructed_volumes[patient_id].keys())]
        sorted_gtv_slices = [patient_gtv_volumes[patient_id][i] for i in sorted(patient_gtv_volumes[patient_id].keys())]

        print(f"Patient {patient_id} - Stacking all image slices in order: {sorted(patient_real_volumes[patient_id].keys())}")

        # Stack slices to form 3D volumes
        real_volume = np.stack(sorted_real_slices, axis=-1)
        reconstructed_volume = np.stack(sorted_reconstructed_slices, axis=-1)
        gtv_volume = np.stack(sorted_gtv_slices, axis=-1)

        # Save 3D volumes as NIfTI files
        real_nifti = nib.Nifti1Image(real_volume, np.eye(4))
        reconstructed_nifti = nib.Nifti1Image(reconstructed_volume, np.eye(4))
        gtv_nifti = nib.Nifti1Image(gtv_volume, np.eye(4))

        nib.save(real_nifti, os.path.join(web_dir, patient_id, f'{patient_id}_real.nii.gz'))
        nib.save(reconstructed_nifti, os.path.join(web_dir, patient_id, f'{patient_id}_reconstructed.nii.gz'))
        nib.save(gtv_nifti, os.path.join(web_dir, patient_id, f'{patient_id}_gtv.nii.gz'))

    # Calculate patient-level statistics
    patient_avg_psnr = {patient: np.mean(values) for patient, values in patient_psnr.items()}
    patient_avg_ssim = {patient: np.mean(values) for patient, values in patient_ssim.items()}
    patient_avg_mse = {patient: np.mean(values) for patient, values in patient_mse.items()}
    patient_avg_mae = {patient: np.mean(values) for patient, values in patient_mae.items()}

    patient_avg_error_map_means = {patient: np.mean(values) for patient, values in patient_error_map_means.items()}
    patient_avg_error_map_stds = {patient: np.mean(values) for patient, values in patient_error_map_stds.items()}

    best_patient = max(patient_avg_psnr, key=patient_avg_psnr.get)
    worst_patient = min(patient_avg_psnr, key=patient_avg_psnr.get)

    for patient, avg_psnr in patient_avg_psnr.items():
        avg_ssim = patient_avg_ssim[patient]
        avg_mse = patient_avg_mse.get(patient, 0)
        avg_mae = patient_avg_mae.get(patient, 0)
        avg_error_map_mean = patient_avg_error_map_means[patient]
        avg_error_map_std = patient_avg_error_map_stds[patient]
        logger.info('Patient %s: Average PSNR = %f, Average SSIM = %f, Average MSE = %f, Average MAE = %f, Avg Error Map Mean = %f, Avg Error Map Std = %f', patient, avg_psnr, avg_ssim, avg_mse, avg_mae, avg_error_map_mean, avg_error_map_std)

    logger.info('Best Patient: %s with Average PSNR = %f', best_patient, patient_avg_psnr[best_patient])
    logger.info('Worst Patient: %s with Average PSNR = %f', worst_patient, patient_avg_psnr[worst_patient])

    with open(os.path.join(web_dir, 'psnr_ssim_error_map_values.txt'), 'w') as f:
        for img_path, psnr_value in psnr_values:
            ssim_value = next(value for path, value in ssim_values if path == img_path)
            mse_value = next(value for path, value in mse_values if path == img_path)
            mae_value = next(value for path, value in mae_values if path == img_path)
            error_map_mean = next(value for path, value in error_map_mean_values if path == img_path)
            error_map_std = next(value for path, value in error_map_std_values if path == img_path)
            f.write(f'Image ({img_path}): PSNR = {psnr_value}, SSIM = {ssim_value}, MSE = {mse_value}, MAE = {mae_value}, Error Map Mean = {error_map_mean}, Error Map Std = {error_map_std}\n')
        f.write('\nAverage PSNR, SSIM, MSE, MAE, and Error Map metrics for each patient:\n')
        for patient, avg_psnr in patient_avg_psnr.items():
            avg_ssim = patient_avg_ssim[patient]
            avg_mse = patient_avg_mse[patient]
            avg_mae = patient_avg_mae[patient]
            avg_error_map_mean = patient_avg_error_map_means[patient]
            avg_error_map_std = patient_avg_error_map_stds[patient]
            f.write(f'Patient {patient}: Average PSNR = {avg_psnr}, Average SSIM = {avg_ssim}, Average MSE = {avg_mse}, Average MAE = {avg_mae}, Avg Error Map Mean = {avg_error_map_mean}, Avg Error Map Std = {avg_error_map_std}\n')
        f.write(f'\nBest Patient: {best_patient} with Average PSNR = {patient_avg_psnr[best_patient]}\n')
        f.write(f'Worst Patient: {worst_patient} with Average PSNR = {patient_avg_psnr[worst_patient]}\n')

    plot_psnr_statistics(patient_avg_psnr, os.path.join(web_dir, 'psnr_statistics.png'))

    webpage.save()  # save the HTML
