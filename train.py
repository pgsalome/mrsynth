import time
import wandb
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    train_dataset = create_dataset(opt)  # create training dataset
    train_dataset_size = len(train_dataset)  # get the number of images in the training dataset
    print('The number of training images = %d' % train_dataset_size)

    opt.phase = 'val'
    val_dataset = create_dataset(opt)  # create validation dataset
    val_dataset_size = len(val_dataset)  # get the number of images in the validation dataset
    print('The number of validation images = %d' % val_dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that displays/saves images and plots
    wandb.init(project='pix2pix', config=opt, name='trial_trainingimages_with_scheduler_earlystopping_plateau')

    total_iters = 0  # the total number of training iterations
    best_psnr = -float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 30
    psnr_threshold = 0.05

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        last_batch_data = None  # Variable to hold the last batch data

        for i, data in enumerate(train_dataset):
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            last_batch_data = data  # Save the current batch data as the last batch

            # # Print the filenames before each iteration
            A_paths = model.image_paths  # Get image paths




            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()


            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / train_dataset_size, losses)
                wandb.log({"G_GAN_train": losses['G_GAN'],
                           "G_L1_train": losses['G_L1'],
                           "D_real_train": losses['D_real'],
                           "D_fake_train": losses['D_fake']})



            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # Log one image from the last batch of the epoch
        if last_batch_data:
            model.set_input(last_batch_data)  # Set the model input to the last batch
            visuals = model.get_current_visuals_train()
            real_A_images = visuals['real_A']
            real_B_images = visuals['real_B']
            fake_B_images = visuals['fake_B']

            def concatenate_images(img_A, img_B, img_fake_B):
                img_A_np = img_A.cpu().detach().numpy().transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
                img_B_np = img_B.cpu().detach().numpy().transpose(1, 2, 0)
                img_fake_B_np = img_fake_B.cpu().detach().numpy().transpose(1, 2, 0)

                img_A_ch1 = img_A_np[..., 0]
                img_A_ch2 = img_A_np[..., 1]
                img_A_ch3 = img_A_np[..., 2]
                img_B_np = img_B_np[..., 0]
                img_fake_B_np = img_fake_B_np[..., 0]

                # Ensure all images have a single channel and same dimensions
                img_A_concat = np.concatenate([img_A_ch1, img_A_ch2,img_A_ch3],axis=1)
                img_concat = np.concatenate([img_A_concat, img_B_np, img_fake_B_np], axis=1)  # Concatenate along width
                return img_concat

            # Select the first image from the last batch
            img_A = real_A_images[0]
            img_B = real_B_images[0]
            img_fake_B = fake_B_images[0]
            img_path = A_paths[0]  # Get the image path for the first image in the batch

            concatenated_img = concatenate_images(img_A, img_B, img_fake_B)
            wandb_img = wandb.Image(concatenated_img, mode='L', caption=f"Epoch {epoch} - Combined Image - {img_path}")
            wandb.log({f"Epoch {epoch} - Combined Image": wandb_img})



        # Perform validation at the end of each epoch
        if epoch % opt.val_freq == 0:
            print(f"Performing validation at the end of epoch {epoch}")
            val_losses = model.validate(val_dataset)
            avg_psnr = val_losses['PSNR']


            # Reduce learning rate if no improvement for `scheduler_patience` epochs
            model.update_learning_rate(avg_psnr)

            # Log validation losses to wandb
            wandb.log({"G_GAN_val": val_losses['G_GAN'],
                       "G_L1_val": val_losses['G_L1'],
                       "D_real_val": val_losses['D_real'],
                       "D_fake_val": val_losses['D_fake'],
                       "PSNR_loss": val_losses['PSNR'],
                       "SSIM_loss": val_losses['SSIM']})


            # Check for improvement
            if avg_psnr > best_psnr or abs(avg_psnr - best_psnr) <= psnr_threshold:
                best_psnr = avg_psnr
                epochs_no_improve = 0  # Reset since there's an improvement
                print(f"New best model saved! PSNR: {avg_psnr:.4f}")
                model.save_networks('best')

                # Log best PSNR to wandb
                wandb.log({"best_PSNR": best_psnr})
            else:
                epochs_no_improve += 1  # Increment since there's no improvement
                print(f"No improvement in PSNR. Number of epochs with no improvement: {epochs_no_improve}")

                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
