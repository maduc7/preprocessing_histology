import numpy as np
from typing import Tuple, List
import utils
import os
import glob
import concurrent.futures
import itertools
import time

def tiles_extractor(magnification: int = 5,
                    ) -> None:
    """

    :param magnification:
    :return:
    """
    img_down_factor = utils.MAGNIFICATION_TO_DOWN_FACT[magnification]

    #utils.create_dir(path_dir=, folder=)


def extract_one_magnification(magnification: int = 5,
                              magnification_down_factor: int = 32
                               ) -> None:
    """

    :param magnification:
    :param magnification_down_factor:
    :return:
    """
    return tiles_extractor(magnification, magnification_down_factor)


if __name__ == '__main__':
    CONFIG_NAME = "../config/config_test.yaml"

    config = utils.load_yaml_config(config_path=CONFIG_NAME)
    print(config)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # process list of files but split across process pool to use more than 1 CPUs
        results = executor.map(extract_one_magnification,
                               config['PARAMETERS']['MAGNIFICATIONS'],
                               itertools.repeat(utils.MAGNIFICATION_TO_DOWN_FACT))
