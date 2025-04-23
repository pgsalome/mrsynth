#
# import os
# from data.base_dataset import BaseDataset, get_params, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
# import torch
#
# class AlignedDataset(BaseDataset):
#     """A dataset class for paired image dataset.
#
#     It assumes that the directory '/path/to/data/train' contains images in the form of {A1, A2, B} concatenated side by side.
#     """
#
#     def __init__(self, opt):
#         """Initialize this dataset class.
#
#         Parameters:
#             opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseDataset.__init__(self, opt)
#         self.dir_ABC = os.path.join(opt.dataroot, opt.phase)  # get the image directory
#         self.ABC_paths = sorted(make_dataset(self.dir_ABC, opt.max_dataset_size))  # get image paths
#         assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
#         self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
#         self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
#
#     def __getitem__(self, index):
#         """Return a data point and its metadata information.
#
#         Parameters:
#             index - - a random integer for data indexing
#
#         Returns a dictionary that contains A (concatenated A1 and A2), B, A_paths and B_paths
#             A (tensor) - - concatenated images in the input domain
#             B (tensor) - - its corresponding image in the target domain
#             A_paths (str) - - image paths
#             B_paths (str) - - image paths (same as A_paths)
#         """
#         # read a image given a random integer index
#         ABC_path = self.ABC_paths[index]
#         ABC = Image.open(ABC_path).convert('RGB')
#
#         # split ABC image into A1, A2, and B
#         w, h = ABC.size
#         w3 = int(w / 3)
#         A1 = ABC.crop((0, 0, w3, h))
#         A2 = ABC.crop((w3, 0, 2*w3, h))
#         B = ABC.crop((2*w3, 0, w, h))
#
#         # apply the same transform to A1, A2, and B
#         transform_params = get_params(self.opt, A1.size)
#         A1_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
#         A2_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
#         B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
#
#         A1 = A1_transform(A1)
#         A2 = A2_transform(A2)
#         B = B_transform(B)
#
#         # concatenate A1 and A2 along the channel dimension
#         A = torch.cat([A1, A2], 0)
#
#         return {'A': A, 'B': B, 'A_paths': ABC_path, 'B_paths': ABC_path}
#
#     def __len__(self):
#         """Return the total number of images in the dataset."""
#         return len(self.ABC_paths)

import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains images in the form of {A1, A2, B} concatenated side by side.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_ABCD = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.ABCD_paths = sorted(make_dataset(self.dir_ABCD, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A (concatenated A1 and A2), B, A_paths and B_paths
            A (tensor) - - concatenated images in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        ABCD_path = self.ABCD_paths[index]
        ABCD = Image.open(ABCD_path).convert('RGB')

        # split ABC image into A1, A2, and B
        w, h = ABCD.size
        w4 = int(w / 4)
        A = ABCD.crop((0, 0, w4, h))
        B = ABCD.crop((w4, 0, 2 * w4, h))
        C = ABCD.crop((2 * w4, 0, 3 * w4, h))
        D = ABCD.crop((3 * w4, 0, w, h))

        # apply the same transform to A, B, C, and D
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        C_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        D_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)
        D = D_transform(D)

        # concatenate A, B, and D along the channel dimension
        input_image = torch.cat([A, B, D], 0)

        return {'A': input_image, 'B': C, 'A_paths': ABCD_path, 'B_paths': ABCD_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABCD_paths)
