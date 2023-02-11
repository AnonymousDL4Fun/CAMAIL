from torchvision import transforms
import torch
import numpy as np
import torch.functional as F
from PIL import Image
from skimage import color

"""
This file provides image normalization paramenters and aumentation methods used
during training and testing

Note: During training, apply Augmentations and then Transformation
      During testing, only apply Transformation 
"""

##############################################
# Custom augmentations
##############################################

class RandomRotationx90(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_params() -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = torch.randint(low=0, high=3, size=[1]).item()  # float(, float(degrees[1])).item())
        return angle * 90

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        angle = self.get_params()

        return F.rotate(img, angle, 2)


class GaussianNoise(torch.nn.Module):

    def forward(self, input_PIL_img):
        """
        Args:
            input_PIL_img: PIL image to add noise on
        Returns:
            output: PIL image with noise. (Current noise std=10)
        """
        np_image = np.array(input_PIL_img)[:, :, 0:3]
        row, col, ch = np_image.shape
        mean = 0
        sigma = 10
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = np.clip(np_image + gauss, 0, 255)
        output = Image.fromarray(noisy.astype(np.uint8))

        return output


class HEDJitter(object):
    # Psuedo Code
    # 1. Change from RGB --> HED space
    # 2. Jitter channels
    # 3. HED--> RGB
    # 4. Blend white region from original image

    # theta determines the amount of jittering per channel in HED --> hematoxylin eosin DAB
    # theta = 0.02 is the defualt

    # 'th' is threshold for blending white background regions from original image. 'th' is in teh range [0,1]
    # default 'th' is 0.9 which corresponds to 0.9*255

    def __init__(self, theta=0.04):
        self.theta = theta
        self.th = 0.9

    def __call__(self, img):
        self.alpha = torch.distributions.uniform.Uniform(1 - self.theta, 1 + self.theta).sample([1, 3])
        self.beta = torch.distributions.uniform.Uniform(-self.theta, self.theta).sample([1, 3])
        img = np.array(img)

        s = color.rgb2hed(img)
        ns = self.alpha * s + self.beta  # perturbations on HED color space
        nimg = color.hed2rgb(ns)

        imin = nimg.min()
        imax = nimg.max()
        rsimg = ((nimg - imin) / (imax - imin) + 1e-6)  # rescale


        rsimg = (255 * rsimg).astype('uint8')

        return Image.fromarray(rsimg)

###########################################################
# Kather Dataset specific transforms and augmentations
###########################################################
trans = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                            ])

aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomVerticalFlip(p=0.5),
                          RandomRotationx90(),
                          transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                          transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                          transforms.GaussianBlur(kernel_size=5),
                          GaussianNoise(),
                          ])

###########################################################
# Camelyon17 Dataset specific transforms and augmentations
###########################################################
trans = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                            ])

aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomVerticalFlip(p=0.5),
                          RandomRotationx90(),
                          HEDJitter(0.004),
                          transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                          transforms.GaussianBlur(kernel_size=3),
                          GaussianNoise(),
                          ])

