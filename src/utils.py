import os
import yaml
import numpy as np
from skimage import filters
from openslide import OpenSlide, open_slide
from skimage.transform import rescale, resize
from PIL import Image
from typing import Tuple

DOWNSAMPLE_FACTOR_TO_LEVEL = {
    1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9, 1024: 10, 2048: 11
}

MAGNIFICATION_TO_DOWN_FACT = {
    40: 1, 20: 2, 10: 4, 5: 8
}


def create_dir(path_dir: str,
               folder: str,
               verbose: bool = False
               ) -> str:
    """
    Create folder if does not exist in path_dir

    :param path_dir: path to directory
    :param folder: name of folder to create
    :param verbose
    :return: complete path to the folder
    """
    new_dir = os.path.join(path_dir, folder)
    if not os.path.exists(new_dir):
        if verbose: print("Creating folder {} in {}".format(folder, path_dir))
        os.makedirs(new_dir)
    return new_dir


def create_folder(path_folder: str,
                  verbose: bool = False
                  ) -> None:
    """

    :param path_folder:
    :param verbose:
    :return:
    """
    if not os.path.exists(path_folder):
        if verbose: print("Creating folder {}".format(path_folder))
        os.makedirs(path_folder)


def load_yaml_config(config_path: str,
                     verbose: bool = False
                     ) -> dict:
    """
    Load a YAML

    :param config_path:
    :param verbose
    :return: python dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if verbose: print("Loading YAML config file: {}".format(config_path))
    return config


def bckgd_tiles(img_np: np.ndarray,
                bckgd_threshold: float = 5e-4,
                verbose: bool = False
                ) -> bool:
    """
    Check if tile if only background (based on histogram of the image and a threshold)

    :param img_np
    :param bckgd_threshold:
    :param verbose
    :return: boolean whether should keep tile or not
    """
    keep_tile = True
    h, _ = np.histogram(img_np, bins=range(0, 260, 50), density=True)
    non_bckgd = np.sum(h[:-1])
    # when smaller than threshold -> only background
    if non_bckgd < bckgd_threshold:
        keep_tile = False

    if verbose: print("Tile is kept: {}".format(keep_tile))
    return keep_tile


def load_openslide_img(img_path: str,
                       verbose: bool = False
                       ) -> OpenSlide:
    """
    load Openslide image

    :param img_path: path to the image
    :param verbose
    :return: openslide image
    """
    if verbose: print("Loading {} Openslide Image.".format(img_path))
    return open_slide(img_path)


def load_openslide_as_pil_img(img_path: str,
                              down_fact: int = 32,
                              verbose: bool = False
                              ) -> Image:
    """
    load PIL image at specific downsample factor

    :param img_path: path to the image
    :param down_fact: factir ti downsample the image
    :param verbose
    :return: PIL image
    """
    if verbose: print("Loading {} image at {} downsample factor.".format(img_path, down_fact))
    img = load_openslide_img(img_path)
    best_level = img.get_best_level_for_downsample(down_fact)
    return img.get_thumbnail(img.level_dimensions[best_level])


def load_pil_img(img_path: str,
                 verbose: bool = False
                 ) -> Image:
    """
    load PIL image

    :param img_path: path to the image
    :param verbose
    :return: PIL image
    """
    if verbose: print("Loading PIL image: {}".format(img_path))
    return Image.open(img_path)


def save_pil_img(pil_img: Image,
                 path_save: str,
                 ext: str = "tiff",
                 verbose: bool = False
                 ) -> None:
    """
    Save image as PIL

    :param pil_img: PIL image
    :param path_save: path to the image to save
    :param ext: extension of the image
    :param verbose
    :return: None
    """
    if verbose: print("Save image as PIL image: {}".format(path_save))
    pil_img.save(path_save, ext)


def save_pil_gray_img(pil_img: Image,
                      path_save: str,
                      ext: str = "tiff",
                      verbose: bool = False
                      ) -> None:
    """
    Save image as PIL

    :param pil_img: PIL image
    :param path_save: path to the image to save
    :param ext: extension of the image
    :param verbose
    :return: None
    """
    if verbose: print("Save image as PIL image: {}".format(path_save))
    pil_img.convert("L").save(path_save, ext)


def pil_to_np(pil_img: Image,
              verbose: bool = False
              ) -> np.ndarray:
    """
    Convert PIL image to numpy array

    :param pil_img: PIL image
    :param verbose
    :return: numpy array
    """
    if verbose: print("Converting PIL image to numpy array.")
    return np.asarray(pil_img)


def np_to_pil(np_img: np.ndarray,
              verbose: bool = False
              ) -> Image:
    """
    Convert numpy array to PIL image

    :param np_img: numpy array
    :param verbose
    :return: PIL image
    """
    if verbose: print("Convert numpy array to PIL image")
    return Image.fromarray(np_img)


def np_rgb_to_np_gray(rgb_np: np.ndarray,
                      verbose: bool = False
                      ) -> np.ndarray:
    """
    convert numpy rgb image to numpy gray image

    :param rgb_np: rgb numpy array
    :param verbose
    :return:  gray numpy array
    """
    if verbose: print("Convert numpy rgb to grayscale image following CCIR 601")
    # for digital formats following CCIR 601: Y= 0.299 R + 0.587 G + 0.114 B
    return np.dot(rgb_np[..., :3], [0.299, 0.587, 0.114])


def pil_rgb_to_pil_gray(rgb_pil: Image,
                        verbose: bool = False
                        ) -> Image:
    """
    convert pil rgb image to numpy gray array

    :param rgb_pil: rgb PIL image
    :param verbose
    :return: gray PIL image
    """
    if verbose: print("Convert PIL rgb image to numpy gray array.")
    return rgb_pil.convert(mode="L")


def get_mpp(openslide_img: OpenSlide,
            verbose: bool = False
            ) -> Tuple[float, float]:
    """
    Get the number of microns per pixel in x an y dimension for the level 0

    :param openslide_img: histology slide
    :param verbose
    :return: mpp-x, mpp-y
    """
    if verbose: print("Get the number of microns per pixel in x and y direction for the level 0.")
    return float(openslide_img.properties["openslide.mpp-x"]), \
           float(openslide_img.properties["openslide.mpp-y"])


def get_dim(openslide_img: OpenSlide,
            verbose: bool = False
            ) -> Tuple[int, int]:
    """
    Get dimension of the slide

    :param openslide_img: histology slide
    :param verbose
    :return: size-x, size-y
    """
    if verbose: print("Get the dimension of the slide.")
    return openslide_img.level_dimensions


def rescale_img(img_np: np.ndarray,
                scale_factor_x: float,
                scale_factor_y: float,
                verbose: bool = False
                ) -> np.ndarray:
    """


    :param img_np: rgb numpy array
    :param scale_factor_x:
    :param scale_factor_y:
    :param verbose
    :return: scaled version of the input image
    """
    if verbose: print("Rescale numpy image by scaling factor x: {} and y: {}".format(scale_factor_x, scale_factor_y))
    return rescale(image=img_np,
                   scale=(scale_factor_x, scale_factor_y),
                   multichannel=True,  # whether the last axis of the image is to be interpreted as multiple channels
                   anti_aliasing=True
                   )


def otsu_thresholding(gray_img: np.ndarray,
                      verbose: bool = False
                      ) -> np.ndarray:
    """
    Apply otsu thresholding

    :param gray_img:
    :param verbose
    :return: mask
    """
    if verbose: print("Otsu thresholding applied to the grayscale image.")
    otsu_thresh = filters.threshold_otsu(gray_img)
    return gray_img < otsu_thresh


def mask_to_img(np_img: np.ndarray,
                mask: np.ndarray,
                verbose: bool = False
                ) -> np.ndarray:
    """
    Apply a mask on an image

    :param np_img: rgb image as numpy array
    :param mask: mask as numpy array
    :param verbose
    :return: mask on image
    """
    if verbose: print("Masking applied on numpy rgb image.")
    return np_img * mask[..., None]


def filter_red(np_img: np.ndarray,
               red_lower_thresh: int,
               green_upper_thresh: int,
               blue_upper_thresh: int
               ) -> np.ndarray:
    # https://github.com/deroneriksson/python-wsi-preprocessing/blob/master/deephistopath/wsi/filter.py
    # 771 to 802 (modified)
    """
    Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
    red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.
    :param np_img: GB image as a NumPy array.
    :param red_lower_thresh: Red channel lower threshold value.
    :param green_upper_thresh: Green channel upper threshold value.
    :param blue_upper_thresh: Blue channel upper threshold value.
    :return: NumPy array representing the mask.
    """
    r = np_img[:, :, 0] > red_lower_thresh
    g = np_img[:, :, 1] < green_upper_thresh
    b = np_img[:, :, 2] < blue_upper_thresh
    return ~(r & g & b)


def filter_red_pen(np_img: np.ndarray
                   ) -> np.ndarray:
    # https://github.com/deroneriksson/python-wsi-preprocessing/blob/master/deephistopath/wsi/filter.py
    # 805 to 833
    """
    Create a mask to filter out red pen marks from a slide.
    :param np_img: RGB image as a NumPy array.
    :return: NumPy array representing the mask.
    """
    result = filter_red(np_img, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
             filter_red(np_img, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
             filter_red(np_img, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
             filter_red(np_img, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
             filter_red(np_img, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
             filter_red(np_img, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
             filter_red(np_img, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
             filter_red(np_img, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
             filter_red(np_img, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
    return result


def filter_green(np_img: np.ndarray,
                 red_upper_thresh: int,
                 green_lower_thresh: int,
                 blue_lower_thresh: int,
                 ) -> np.ndarray:
    # https://github.com/deroneriksson/python-wsi-preprocessing/blob/master/deephistopath/wsi/filter.py
    # 836 to 869
    """

    :param np_img: RGB image as a NumPy array.
    :param red_upper_thresh: Red channel upper threshold value.
    :param green_lower_thresh: Green channel lower threshold value.
    :param blue_lower_thresh: Blue channel lower threshold value.
    :return: NumPy array representing the mask.
    """
    r = np_img[:, :, 0] < red_upper_thresh
    g = np_img[:, :, 1] > green_lower_thresh
    b = np_img[:, :, 2] > blue_lower_thresh
    return ~(r & g & b)


def filter_green_pen(np_img: np.ndarray,
                     ) -> np.ndarray:
    # https://github.com/deroneriksson/python-wsi-preprocessing/blob/master/deephistopath/wsi/filter.py
    # 872 to 906
    """
    Create a mask to filter out green pen marks from a slide.
    :param np_img: RGB image as a NumPy array.
    :return: NumPy array representing the mask.
    """
    result = filter_green(np_img, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
             filter_green(np_img, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
             filter_green(np_img, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
             filter_green(np_img, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
             filter_green(np_img, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
             filter_green(np_img, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
             filter_green(np_img, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
             filter_green(np_img, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
             filter_green(np_img, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
             filter_green(np_img, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
             filter_green(np_img, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
             filter_green(np_img, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
             filter_green(np_img, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
             filter_green(np_img, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
             filter_green(np_img, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
    return result


def filter_blue(np_img: np.ndarray,
                red_upper_thresh: int,
                green_upper_thresh: int,
                blue_lower_thresh: int
                ) -> np.ndarray:
    # https://github.com/deroneriksson/python-wsi-preprocessing/blob/master/deephistopath/wsi/filter.py
    # 909 to 940
    """
    Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
    red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

    :param np_img: RGB image as a NumPy array.
    :param red_upper_thresh: Red channel upper threshold value.
    :param green_upper_thresh:  Green channel upper threshold value.
    :param blue_lower_thresh: Blue channel lower threshold value.
    :return: NumPy array representing the mask.
    """
    r = np_img[:, :, 0] < red_upper_thresh
    g = np_img[:, :, 1] < green_upper_thresh
    b = np_img[:, :, 2] > blue_lower_thresh
    return ~(r & g & b)


def filter_blue_pen(np_img: np.ndarray
                    ) -> np.ndarray:
    # https://github.com/deroneriksson/python-wsi-preprocessing/blob/master/deephistopath/wsi/filter.py
    # 943 to 974
    """
    Create a mask to filter out blue pen marks from a slide.
    :param np_img: RGB image as a NumPy array.
    :return: NumPy array representing the mask.
    """
    result = filter_blue(np_img, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
             filter_blue(np_img, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
             filter_blue(np_img, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
             filter_blue(np_img, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
             filter_blue(np_img, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
             filter_blue(np_img, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
             filter_blue(np_img, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
             filter_blue(np_img, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
             filter_blue(np_img, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
             filter_blue(np_img, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
             filter_blue(np_img, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
             filter_blue(np_img, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
    return result


def filter_grays(np_img: np.ndarray,
                 tolerance: int = 15
                 ) -> np.ndarray:
    # https://github.com/deroneriksson/python-wsi-preprocessing/blob/master/deephistopath/wsi/filter.py
    # 977 to 1005
    """
    Create a mask to filter out pixels where the red, green, and blue channel values are similar.
    :param np_img: RGB image as a NumPy array.
    :param tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    :return: NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """
    np_img = np_img.astype(np.int)
    rg_diff = abs(np_img[:, :, 0] - np_img[:, :, 1]) <= tolerance
    rb_diff = abs(np_img[:, :, 0] - np_img[:, :, 2]) <= tolerance
    gb_diff = abs(np_img[:, :, 1] - np_img[:, :, 2]) <= tolerance

    return ~(rg_diff & rb_diff & gb_diff)
