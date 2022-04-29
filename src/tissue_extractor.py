import utils
import numpy as np
import os
import glob
import time
import concurrent.futures
import itertools
import yaml


def process_one_image(img_path: str,
                      config: yaml
                      ) -> None:
    """
    Extract the tissue of one image by doing the following steps:
    1) detect part of the images that are black, blue, red or green (which could be pen drawing)
    2) replace those pen drawing with white
    3) use Otsu thresholding to detect tissue
    4) remove columns and rows that are mainly white
    5) save tissue cropped image

    :param img_path: path to the image
    :param config: config file
    :return:
    """
    # name image
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(img_name)

    saving_ext_format = config['DATA']['EXTENSION_SAVE']
    save_img = config['DATA']['SAVE_PLOT_IMG']
    save_path_plot = config['DATA']['DATA_SAVE_PLOT']
    verbose = config['VERBOSE']

    # 1) save image as baseline in data/plot/raw/
    pil_rgb = utils.load_openslide_as_pil_img(img_path=img_path,
                                              down_fact=config['PARAMETERS']['DOWNSAMPLE_FACTOR'],
                                              verbose=verbose)
    np_rgb = utils.pil_to_np(pil_rgb)
    if save_img:
        path_save_raw_img = save_path_plot + "raw/"
        utils.create_folder(path_save_raw_img, verbose)
        utils.save_pil_img(utils.np_to_pil(np_rgb),
                           path_save_raw_img + img_name,
                           config['DATA']['EXTENSION_SAVE'],
                           verbose=verbose
                           )

    # detect gray pixels (also black)
    np_rgb_no_gray = utils.filter_grays(np_rgb)
    if save_img:
        path_save_filter_img = save_path_plot + "filter/"
        utils.create_folder(path_save_filter_img, verbose)
        utils.save_pil_img(utils.np_to_pil(np_rgb_no_gray),
                           path_save_filter_img + img_name + "_no_gray",
                           config['DATA']['EXTENSION_SAVE'],
                           verbose=verbose
                           )
    # detect blue pen
    np_rgb_no_blue = utils.filter_blue_pen(np_rgb)
    if save_img:
        path_save_filter_img = save_path_plot + "filter/"
        utils.create_folder(path_save_filter_img, verbose)
        utils.save_pil_img(utils.np_to_pil(np_rgb_no_blue),
                           path_save_filter_img + img_name + '_no_blue',
                           config['DATA']['EXTENSION_SAVE'],
                           verbose=verbose
                           )
    # detect green pen
    np_rgb_no_green = utils.filter_green_pen(np_rgb)
    if save_img:
        path_save_filter_img = save_path_plot + 'filter/'
        utils.create_folder(path_save_filter_img, verbose)
        utils.save_pil_img(utils.np_to_pil(np_rgb_no_green),
                           path_save_filter_img + img_name + '_no_green',
                           config['DATA']['EXTENSION_SAVE'],
                           verbose=verbose
                           )
    # detect red pen
    np_rgb_no_red = utils.filter_red_pen(np_rgb)
    if save_img:
        path_save_filter_img = save_path_plot + "filter/"
        utils.create_folder(path_save_filter_img, verbose)
        utils.save_pil_img(utils.np_to_pil(np_rgb_no_red),
                           path_save_filter_img + img_name + '_no_red',
                           config['DATA']['EXTENSION_SAVE'],
                           verbose=verbose
                           )

    # 2) crop tissue based on Otsu thresholding and save image in data/plot/crop/
    np_gray = utils.np_rgb_to_np_gray(np_rgb)
    if save_img:
        path_save_gray_img = save_path_plot + "gray/"
        utils.create_folder(path_save_gray_img, verbose)
        utils.save_pil_gray_img(utils.np_to_pil(np_gray),
                                path_save_gray_img + img_name,
                                config['DATA']['EXTENSION_SAVE'],
                                verbose=verbose
                                )

    # gray image without the red, blue, green and black pen marks
    # replace pixels with white pixels
    np_gray = np.where(np_rgb_no_gray, np_gray, 255)
    np_gray = np.where(np_rgb_no_blue, np_gray, 255)
    np_gray = np.where(np_rgb_no_green, np_gray, 255)
    np_gray = np.where(np_rgb_no_red, np_gray, 255)
    if save_img:
        path_save_gray_img = save_path_plot + "gray/"
        utils.create_folder(path_save_gray_img, verbose)
        utils.save_pil_gray_img(utils.np_to_pil(np_gray),
                                path_save_gray_img + img_name + "_no_pen",
                                config['DATA']['EXTENSION_SAVE'],
                                verbose=verbose
                                )

    # otsu thresholding on the grayscale images that has no black or pen marks
    otsu_mask = utils.otsu_thresholding(np_gray)
    if save_img:
        path_save_otsu_img = save_path_plot + "otsu/"
        utils.create_folder(path_save_otsu_img, verbose)
        utils.save_pil_img(utils.np_to_pil(otsu_mask),
                           path_save_otsu_img + img_name,
                           config['DATA']['EXTENSION_SAVE'],
                           verbose=verbose
                           )

    new_np_img_avg_col = np.mean(otsu_mask, axis=0)
    new_np_img_avg_row = np.mean(otsu_mask, axis=1)

    otsu_threshold = config['PARAMETERS']['OTSU']['THRESHOLD']

    # get rows where mostly white background
    idx_del_rows = np.where(new_np_img_avg_row < otsu_threshold)
    # get columns where mostly white background
    idx_del_cols = np.where(new_np_img_avg_col < otsu_threshold)

    if save_img:
        # delete rows where mostly white background
        new_np_img = np.delete(np_rgb, np.where(new_np_img_avg_row < otsu_threshold), axis=0)
        # delete columns where mostly white background
        new_np_img = np.delete(new_np_img, np.where(new_np_img_avg_col < otsu_threshold), axis=1)
        path_save_crop_img = save_path_plot + "crop/"
        utils.create_folder(path_save_crop_img, verbose)
        save_img_path = path_save_crop_img + img_name
        utils.save_pil_img(utils.np_to_pil(new_np_img), save_img_path, saving_ext_format)

    # create a mask of where there is tissue and where it is background
    save_mask = config['DATA']['DATA_SAVE_TISSUE_MASKS']
    utils.create_folder(save_mask, verbose)
    save_crop_img_mask_path = save_mask + img_name + '_df_' + str(config['PARAMETERS']['DOWNSAMPLE_FACTOR'])

    # remove small objects
    idx_del_rows, idx_del_cols = utils.filter_small_part(idx_row_np=idx_del_rows,
                                                         idx_col_np=idx_del_cols,
                                                         img_np_shape=otsu_mask.shape,
                                                         tolerance=config['PARAMETERS']['SMALL_OBJ']['TOLERANCE'])
    # keep small objects
    idx_del_rows, idx_del_cols = utils.fill_small_part(bool_row_np=idx_del_rows,
                                                       bool_col_np=idx_del_cols,
                                                       tolerance=config['PARAMETERS']['SMALL_OBJ']['TOLERANCE'])

    if save_img:
        # delete rows where mostly white background
        new_np_img = np.delete(np_rgb, np.where(idx_del_rows == False), axis=0)
        # delete columns where mostly white background
        new_np_img = np.delete(new_np_img, np.where(idx_del_cols == False), axis=1)
        path_save_crop_img = save_path_plot + "crop/"
        utils.create_folder(path_save_crop_img, verbose)
        save_img_path = path_save_crop_img +img_name+'_final'
        utils.save_pil_img(utils.np_to_pil(new_np_img), save_img_path, saving_ext_format)

    mask = np.ones_like(otsu_mask)
    mask[~idx_del_rows, :] = 0
    mask[:, ~idx_del_cols] = 0
    assert mask.shape == np_gray.shape

    utils.save_pil_img(utils.np_to_pil(mask), save_crop_img_mask_path, saving_ext_format)
    print("save cropping mask as: ", save_crop_img_mask_path)


def slide_crop_single_processing(config: yaml
                                 ) -> None:
    """
    crop one single image at a time

    :param config:
    :return:
    """
    img_dir_path = config["DATA"]["DATA_SOURCE"]
    # loop over all images from the class
    for img_path in glob.glob(img_dir_path+'/*'):
        process_one_image(img_path, config)


def slide_crop_multi_processing(config: yaml
                                ) -> None:
    """
    crop multiple images at a time using concurrent.futures

    :param config:
    :return:
    """
    img_dir_path = config['DATA']['DATA_SOURCE']
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # list of files to process
        img_paths = glob.glob(img_dir_path+'/*')
        # process list of files but split across process pool to use all CPUs
        executor.map(process_one_image, img_paths, itertools.repeat(config))


if __name__ == '__main__':
    print("----------------------------------")
    print("|      Cropping the slides       |")
    print("----------------------------------")
    CONFIG_NAME = '../config/config_test.yaml'

    config = utils.load_yaml_config(config_path=CONFIG_NAME)
    print(config)

    # process data in parallel
    multi_process = config['MULTI_PROCESSING']

    start = time.time()

    if multi_process:
        slide_crop_multi_processing(config)
    else:
        slide_crop_single_processing(config)

    total_time = time.time() - start
    print('-------------------------')
    print('  End of slides cropping ')
    print('         {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('-------------------------')
    print('')
