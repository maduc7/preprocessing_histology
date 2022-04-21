import utils
import glob

## 1) save image as baseline in data/plot/raw/
CONFIG_NAME = "../config/config_test.yaml"

config = utils.load_yaml_config(config_path=CONFIG_NAME)
print(config)

verbose = config['VERBOSE']

path_save_raw_img = "../data/plot/raw/"

utils.create_folder(config['DATA']['DATA_SAVE'], verbose)
utils.create_folder(path_save_raw_img, verbose)

for img_path in glob.glob(config['DATA']['DATA_SOURCE'] + '/*'):
    pil_rgb = utils.load_pil_img(img_path, verbose)
    utils.save_pil_img(pil_rgb, path_save_raw_img, config['DATA']['EXTENSION_SAVE'], verbose)
## 2) crop tissue based on Otsu thresholding and save image in data/plot/crop/
utils.create_folder("../data/plot/crop/", verbose)
## 3) extract tiles and save them in data/tiles/