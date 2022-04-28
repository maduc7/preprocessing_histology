import numpy as np
from typing import Tuple, List
import utils
import os
import glob
import concurrent.futures
import itertools
import time
import yaml


def extract_one_magnification(img_path: str,
                              config: yaml,
                              magnification: int = 5
                               ) -> None:
    """

    :param img_path:
    :param magnification:
    :param magnification_down_factor:
    :return:
    """
    img_down_factor = utils.MAGNIFICATION_TO_DOWN_FACT[magnification]
    img_pil = utils.load_openslide_as_pil_img(img_path, img_down_factor)
    print("img pil", img_pil)
    img_np = utils.pil_to_np(img_pil)
    print("img np size", img_np.shape)

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(img_name)

    save_path_tiles = config['DATA']['DATA_SAVE_TILES']
    save_path_mag = save_path_tiles + str(magnification) + "/"

    # get mask
    mask_down_factor = config['PARAMETERS']['DOWNSAMPLE_FACTOR']
    mask_to_img_down_factor = int(mask_down_factor/img_down_factor)
    mask_tile_size = int(config['PARAMETERS']['TILE_SIZE']/mask_to_img_down_factor)
    mask_path = config['DATA']['DATA_SAVE_TISSUE_MASKS']+img_name+'_df_'+str(mask_down_factor)

    if not os.path.exists(mask_path):
        print("xxxxxxx------------xxxxxxx")
        print("!! No mask for: "+img_name+"!!")
        print("xxxxxxx------------xxxxxxx")
    else:
        mask_np = utils.pil_to_np(utils.load_pil_img(mask_path))
        img_np = utils.crop_img_from_mask(img_np, mask_np, mask_to_img_down_factor)
        shape_mask_r = img_np.shape[0] // mask_to_img_down_factor
        shape_mask_c = img_np.shape[1] // mask_to_img_down_factor

        # number of x and y tiles
        num_row_tiles, num_col_tiles = shape_mask_r // mask_tile_size, shape_mask_c // mask_tile_size
        nb_max_patches = num_row_tiles * num_col_tiles
        print("Total nb of tiles {} (rows: {}, cols: {}): ".format(nb_max_patches, num_row_tiles, num_col_tiles))
        save_path_img = save_path_mag + "/" + img_name

        extract_random_patches(save_path_mag, img_np, config, nb_max_patches, num_col_tiles, img_name)


def extract_random_patches(save_path_mag: str,
                           img_np: np.ndarray,
                           config: yaml,
                           nb_max_patches: int,
                           nb_col: int,
                           img_name: str
                           ) -> None:
    verbose = config['VERBOSE']
    nb_tiles = config['PARAMETERS']['NB_TILES']
    nb_idx = np.minimum(nb_tiles, nb_max_patches)

    path_save_bckg = save_path_mag + "bckg/"
    path_save_tissue = save_path_mag + "tissue/"
    utils.create_folder(path_save_bckg, verbose)
    utils.create_folder(path_save_tissue, verbose)

    if verbose:
        print("nb of idx to select: ", nb_idx)
    np.random.seed(config['SEED_VALUE'])
    random_idx = np.random.choice(nb_max_patches, size=nb_max_patches, replace=False)

    tile_size = config['PARAMETERS']['TILE_SIZE']
    i = 0
    idx = 0
    while i < nb_idx and idx < nb_max_patches:
        print("nb patch: ", i, idx)
        # get the corresponding row and column from the index
        r = int(random_idx[idx] / nb_col)
        c = random_idx[idx] % nb_col
        img_tile = img_np[r * tile_size:(r + 1) * tile_size, c * tile_size:(c + 1) * tile_size]

        # check if random_idx[idx] is not background and save the tile
        if utils.bckgd_tiles(img_tile,
                             bckgd_threshold=config['PARAMETERS']['BACKGROUND']['THRESHOLD'],
                             verbose=verbose):
            i += 1
            save_path = path_save_tissue + img_name + "_" + str(i)
            utils.save_pil_img(pil_img=utils.np_to_pil(img_tile),
                               path_save=save_path,
                               ext=config['DATA']['EXTENSION_SAVE'])
        else:
            # save tiles in background plot to check that this works well!!!
            print("This is a background tile")
            save_path = path_save_bckg + img_name + "_" + str(idx)
            utils.save_pil_img(pil_img=utils.np_to_pil(img_tile),
                               path_save=save_path,
                               ext=config['DATA']['EXTENSION_SAVE'])
        idx += 1


def slide_extract_single_processing(config: yaml
                                    ) -> None:
    """
    extract one single image at a time and one single magnification

    :param config:
    :return:
    """
    img_dir_path = config["DATA"]["DATA_SOURCE"]
    magnifications = config['PARAMETERS']['MAGNIFICATIONS']
    # loop over all images from the class
    for img_path in glob.glob(img_dir_path+'/*'):
        # loop over all magnification
        for mag in magnifications:
            extract_one_magnification(img_path=img_path,
                                      config=config,
                                      magnification=mag)


def slide_extract_multiple_processing(config: yaml
                                      ) -> None:
    """
    extract multiple magnifications at a time using concurrent.futures

    :param config:
    :return:
    """
    img_dir_path = config['DATA']['DATA_SOURCE']
    magnifications = config['PARAMETERS']['MAGNIFICATIONS']
    # loop over all images from the class
    for img_path in glob.glob(img_dir_path + '/*'):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # process list of files but split across process pool to use all CPUs
            executor.map(extract_one_magnification,
                         itertools.repeat(img_path),
                         itertools.repeat(config),
                         magnifications)


if __name__ == '__main__':
    print("----------------------------------")
    print("|      Extracting tiles at       |")
    print("|     specific magnifications    |")
    print("----------------------------------")

    CONFIG_NAME = "../config/config_test.yaml"
    config = utils.load_yaml_config(config_path=CONFIG_NAME)
    print(config)

    # process data in parallel
    multi_process = config['MULTI_PROCESSING']

    start = time.time()

    # loop over all images from the class
    if multi_process:
        slide_extract_multiple_processing(config)
    else:
        slide_extract_single_processing(config)

    total_time = time.time() - start
    print('-------------------------')
    print('  End of slides cropping ')
    print('         {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('-------------------------')
    print('')
