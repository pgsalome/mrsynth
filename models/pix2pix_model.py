import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2)


def calculate_ssim(img1, img2):
    # Ensure the images are in float format
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Get the data range (for normalized images [0, 1], data_range is 1.0)
    data_range = img1.max() - img1.min()

    # Get the minimum dimension of the images
    min_dim = min(img1.shape[0], img1.shape[1])

    # Set win_size to be the largest odd number less than or equal to min_dim
    win_size = min(7, min_dim // 2 * 2 + 1)

    # Handle the case where images are smaller than 7x7
    if min_dim < 7:
        win_size = min_dim // 2 * 2 + 1  # Make sure win_size is odd and <= min_dim

    return structural_similarity(img1, img2, multichannel=True, win_size=win_size, data_range=data_range, channel_axis=-1)


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt, num_input_channels):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)

        self.netG = networks.define_G(num_input_channels*opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(num_input_channels*opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        out = self.fake_B
        return out

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        #print("Optimizing parameters...")
        self.forward()  # compute fake images: G(A)

        # Update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
       # print(f"Discriminator loss: {self.loss_D.item()}")
        self.optimizer_D.step()  # update D's weights
        #print("Discriminator updated.")

        # Update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        #print(f"Generator loss: {self.loss_G.item()}")
        self.optimizer_G.step()  # update G's weights
       # print("Generator updated.")

    def train(self):
        """Set model to training mode"""
        print("Entering training mode...")
        self.netG.train()
        self.netD.train()



    def validate(self, val_dataset):
        """Run validation"""
        self.eval()  # Set model to evaluation mode
        total_loss_G_GAN = 0
        total_loss_G_L1 = 0
        total_loss_D_real = 0
        total_loss_D_fake = 0
        total_psnr = 0
        total_ssim = 0
        num_batches = 0

        with torch.no_grad():
            for i, data in enumerate(val_dataset):
                self.set_input(data)
                self.forward()
                self.compute_visuals()

                # Calculate GAN loss for the generator
                fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                pred_fake = self.netD(fake_AB)
                loss_G_GAN = self.criterionGAN(pred_fake, True)
                total_loss_G_GAN += loss_G_GAN.item()

                # Calculate L1 loss for the generator
                loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
                total_loss_G_L1 += loss_G_L1.item()

                # Calculate GAN loss for the discriminator
                # Fake; stop backprop to the generator by detaching fake_B
                fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                pred_fake = self.netD(fake_AB.detach())
                loss_D_fake = self.criterionGAN(pred_fake, False)
                total_loss_D_fake += loss_D_fake.item()

                # Real
                real_AB = torch.cat((self.real_A, self.real_B), 1)
                pred_real = self.netD(real_AB)
                loss_D_real = self.criterionGAN(pred_real, True)
                total_loss_D_real += loss_D_real.item()

                # Calculate PSNR and SSIM
                fake_B_np = self.fake_B.cpu().numpy().transpose(0, 2, 3, 1)
                real_B_np = self.real_B.cpu().numpy().transpose(0, 2, 3, 1)

                batch_psnr = 0
                batch_ssim = 0

                for fake_img, real_img in zip(fake_B_np, real_B_np):
                    psnr =calculate_psnr(fake_img, real_img)
                    ssim = calculate_ssim(fake_img, real_img)
                    #print(f"PSNR: {psnr}, SSIM: {ssim}")

                    batch_psnr += psnr
                    batch_ssim += ssim
                avg_batch_psnr = batch_psnr / len(fake_B_np)
                avg_batch_ssim = batch_ssim / len(fake_B_np)
                #print(f"Batch PSNR: {avg_batch_psnr}, Batch SSIM: {avg_batch_ssim}")
                total_psnr += avg_batch_psnr
                total_ssim += avg_batch_ssim
                num_batches += 1

        avg_loss_G_GAN = total_loss_G_GAN / num_batches
        avg_loss_G_L1 = total_loss_G_L1 / num_batches
        avg_loss_D_real = total_loss_D_real / num_batches
        avg_loss_D_fake = total_loss_D_fake / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches

        # Log average losses
        print(
            f"Validation: Avg G_GAN Loss: {avg_loss_G_GAN}, Avg G_L1 Loss: {avg_loss_G_L1}, Avg D_real Loss: {avg_loss_D_real}, Avg D_fake Loss: {avg_loss_D_fake}, Avg PSNR: {avg_psnr}, Avg SSIM: {avg_ssim}")

        self.train()  # Set model back to training mode

        # Return validation losses for early stopping or model checkpointing
        return {'G_GAN': avg_loss_G_GAN, 'G_L1': avg_loss_G_L1, 'D_real': avg_loss_D_real, 'D_fake': avg_loss_D_fake, 'PSNR': avg_psnr, 'SSIM': avg_ssim}

