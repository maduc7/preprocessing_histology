import utils
import numpy as np
import os
import glob
import time
import concurrent.futures
import itertools


def process_one_image(img_path: str,
                      save_path: str,
                      threshold: float = 0.05,
                      ) -> None:
    # name image
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(img_name)

    pil_rgb = utils.load_pil_img(img_path)
    np_rgb = utils.pil_to_np(pil_rgb)
    np_gray = utils.np_rgb_to_np_gray(np_rgb)

    mask = utils.otsu_thresholding(np_gray)

    new_np_img_avg_col = np.mean(mask, axis=0)
    new_np_img_avg_row = np.mean(mask, axis=1)

    idx_non_zero_col_start = (new_np_img_avg_col > threshold).argmax(axis=0)
    idx_non_zero_row_start = (new_np_img_avg_row > threshold).argmax(axis=0)

    idx_non_zero_col_end = np.max(np.nonzero(new_np_img_avg_col > threshold))
    idx_non_zero_row_end = np.max(np.nonzero(new_np_img_avg_row > threshold))

    new_np_img = np_rgb[idx_non_zero_row_start:idx_non_zero_row_end, idx_non_zero_col_start:idx_non_zero_col_end, :]

    save_img_path = save_path + img_name
    utils.save_pil_img(utils.np_to_pil(new_np_img), save_img_path, saving_ext_format)

    print("save new slide as: ", save_img_path)


def slide_crop_single_processing(img_dir_path: str,
                                 save_path: str,
                                 threshold: float = 0.05,
                                 saving_ext_format: str = "tiff"
                                 ) -> None:
    # loop over all images from the class
    for img_path in glob.glob(img_dir_path + "/*"):
        process_one_image(img_path, save_path, threshold, saving_ext_format)


def slide_crop_multi_processing(img_dir_path: str,
                                save_path: str,
                                threshold: float = 0.05,
                                saving_ext_format: str = "tiff"
                                ) -> None:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # list of files to process
        img_paths = glob.glob(img_dir_path+"/*")
        # process list of files but split across process pool to use all CPUs
        executor.map(process_one_image, img_paths, itertools.repeat(save_path), itertools.repeat(threshold),
                     itertools.repeat(saving_ext_format))


if __name__ == '__main__':
    print("----------------------------------")
    print("|      Cropping the slides       |")
    print("----------------------------------")
    CONFIG_NAME = "../config/config_test.yaml"

    config = utils.load_yaml_config(config_path=CONFIG_NAME)
    print(config)

    ##########################################
    #       Parameters initialization        #
    ##########################################
    # process data in parallel
    multi_process = True

    # threshold of average value of the mask to exclude
    threshold = 0.05

    img_dir_paths = config['DATA']['DATA_SOURCE']
    save_path_dir = "../data/tissue_crop/"
    saving_ext_format = config['DATA']['EXTENSION_SAVE']

    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)

    start = time.time()

    if multi_process:
        slide_crop_multi_processing(img_dir_paths, save_path_dir, threshold, saving_ext_format)
    else:
        slide_crop_single_processing(img_dir_paths, save_path_dir, threshold, saving_ext_format)

    total_time = time.time() - start
    print("-------------------------")
    print("  End of slides cropping ")
    print("         {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60))
    print("-------------------------")
    print("")