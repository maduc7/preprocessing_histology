import utils
import glob
import os
import numpy as np


CONFIG_NAME = "../config/config_test.yaml"

config = utils.load_yaml_config(config_path=CONFIG_NAME)
print(config)

verbose = config['VERBOSE']

path_save_raw_img = "../data/plot/raw/"
path_save_gray_img = "../data/plot/gray/"
path_save_crop_img = "../data/plot/crop/"
path_save_otsu_img = "../data/plot/otsu/"
path_save_filter_img = "../data/plot/filter/"
down_factor_save = 64
otsu_threshold = config['PARAMETERS']['OTSU']['THRESHOLD']

utils.create_folder(config['DATA']['DATA_SAVE'], verbose)
utils.create_folder(path_save_raw_img, verbose)
utils.create_folder(path_save_gray_img, verbose)
utils.create_folder(path_save_crop_img, verbose)
utils.create_folder(path_save_otsu_img, verbose)
utils.create_folder(path_save_filter_img, verbose)

for img_path in glob.glob(config['DATA']['DATA_SOURCE'] + '/*'):
    ## 1) save image as baseline in data/plot/raw/
    pil_rgb = utils.load_openslide_as_pil_img(img_path=img_path, down_fact=down_factor_save, verbose=verbose)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    np_rgb = utils.pil_to_np(pil_rgb)
    utils.save_pil_img(utils.np_to_pil(np_rgb),
                       path_save_raw_img + img_name,
                       config['DATA']['EXTENSION_SAVE'],
                       verbose=verbose
                       )

    np_rgb_no_gray = utils.filter_grays(np_rgb)
    utils.save_pil_img(utils.np_to_pil(np_rgb_no_gray),
                       path_save_filter_img + img_name + "_no_gray",
                       config['DATA']['EXTENSION_SAVE'],
                       verbose=verbose
                       )

    np_rgb_no_blue = utils.filter_blue_pen(np_rgb)
    utils.save_pil_img(utils.np_to_pil(np_rgb_no_blue),
                       path_save_filter_img + img_name + "_no_blue",
                       config['DATA']['EXTENSION_SAVE'],
                       verbose=verbose
                       )

    np_rgb_no_green = utils.filter_green_pen(np_rgb)
    utils.save_pil_img(utils.np_to_pil(np_rgb_no_green),
                       path_save_filter_img + img_name + "_no_green",
                       config['DATA']['EXTENSION_SAVE'],
                       verbose=verbose
                       )

    np_rgb_no_red = utils.filter_red_pen(np_rgb)
    utils.save_pil_img(utils.np_to_pil(np_rgb_no_red),
                       path_save_filter_img + img_name + "_no_red",
                       config['DATA']['EXTENSION_SAVE'],
                       verbose=verbose
                       )

    ## 2) crop tissue based on Otsu thresholding and save image in data/plot/crop/
    #pil_rgb = utils.load_pil_img(img_path)
    np_gray = utils.np_rgb_to_np_gray(np_rgb)
    utils.save_pil_gray_img(utils.np_to_pil(np_gray),
                            path_save_gray_img + img_name,
                            config['DATA']['EXTENSION_SAVE'],
                            verbose=verbose
                            )
    # gray image without the red, blue, green and black pen drawing
    np_gray = np.where(np_rgb_no_gray, np_gray, 255)
    np_gray = np.where(np_rgb_no_blue, np_gray, 255)
    np_gray = np.where(np_rgb_no_green, np_gray, 255)
    np_gray = np.where(np_rgb_no_red, np_gray, 255)
    utils.save_pil_gray_img(utils.np_to_pil(np_gray),
                            path_save_gray_img + img_name + "_no_pen",
                            config['DATA']['EXTENSION_SAVE'],
                            verbose=verbose
                            )

    mask = utils.otsu_thresholding(np_gray)
    utils.save_pil_img(utils.np_to_pil(mask),
                       path_save_otsu_img + img_name,
                       config['DATA']['EXTENSION_SAVE'],
                       verbose=verbose
                       )

    new_np_img_avg_col = np.mean(mask, axis=0)
    new_np_img_avg_row = np.mean(mask, axis=1)

    # delete rows where only white background
    new_np_img = np.delete(np_rgb, np.where(new_np_img_avg_row < otsu_threshold), axis=0)
    # delete columns where only white background
    new_np_img = np.delete(new_np_img, np.where(new_np_img_avg_col < otsu_threshold), axis=1)

    utils.save_pil_img(utils.np_to_pil(new_np_img),
                       path_save_crop_img + img_name,
                       config['DATA']['EXTENSION_SAVE'],
                       verbose=verbose)

    ## 3) extract tiles and save them in data/tiles/