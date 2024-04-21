import numpy as np
from typing import List, Tuple
import cv2

t_image_list = List[np.array]
t_str_list = List[str]
t_image_triplet = Tuple[np.array, np.array, np.array]


def show_images(images: t_image_list, names: t_str_list) -> None:
    """Shows one or more images at once.

    Displaying a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        names: A list of strings that will appear as the window titles for each image
    
    Returns:
        None
    """
    for name, img in zip(names, images):
        cv2.imshow(name, img)
    
        cv2.waitKey(0)

def save_images(images: t_image_list, filenames: t_str_list, **kwargs) -> None:
    """Saves one or more images at once.

    Saving a single image can be done by putting it in a list.
    If the paths have directories, they must already exist.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        filenames: A list of strings where each respective file will be created

    Returns:
        None
    """
    for img, filename in zip(images, filenames):
        cv2.imwrite(filename, img)

def scale_down(image: np.array) -> np.array:
    """Returns an image half the size of the original.

    Args:
        image: A numpy array with an opencv image

    Returns:
        A numpy array with an opencv image half the size of the original image
    """
    height, width = image.shape[:2]


    new_height = height // 2
    new_width = width // 2

    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image
def separate_channels(colored_image: np.array) -> t_image_triplet:
    """Takes an BGR color image and splits it three images.

    Args:
        colored_image: an numpy array sized [HxWxC] where the channels are in BGR (Blue, Green, Red) order

    Returns:
        A tuple with three BGR images the first one containing only the Blue channel active, the second one only the
        green, and the third one only the red.
    """
    
    blue_channel = colored_image[:, :, 0]  # Blue channel
    green_channel = colored_image[:, :, 1]  # Green channel
    red_channel = colored_image[:, :, 2]  # Red channel
    blue_image = np.zeros_like(colored_image)
    green_image = np.zeros_like(colored_image)
    red_image = np.zeros_like(colored_image)

    # Assign each channel to the corresponding image
    blue_image[:, :, 0] = blue_channel
    green_image[:, :, 1] = green_channel
    red_image[:, :, 2] = red_channel
    return blue_image, green_image, red_image