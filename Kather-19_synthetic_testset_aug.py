import PIL.Image
import numpy as np
from torchvision import transforms

#Dummy image as placeholder
image = PIL.Image.fromarray(np.zeros(224, 224, 3))

# For generation StainAug set
image = transforms.functional.adjust_hue(image, 0.15)

# For generation BlurAug set
image = transforms.functional.gaussian_blur(image, 5, 10)

# For generation MagAug set
image = transforms.functional.resized_crop(image, 45, 45, 134, 134, 224)