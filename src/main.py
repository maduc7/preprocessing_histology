import utils
import tissue_extractor
import time

print("----------------------------------")
print("|      Cropping the slides       |")
print("----------------------------------")
CONFIG_NAME = "../config/config_test.yaml"

config = utils.load_yaml_config(config_path=CONFIG_NAME)
print(config)

# process data in parallel
multi_process = config['MULTI_PROCESSING']

verbose = config['VERBOSE']

start = time.time()

if multi_process:
    tissue_extractor.slide_crop_multi_processing(config)
else:
    tissue_extractor.slide_crop_single_processing(config)

total_time = time.time() - start
print('-------------------------')
print('  End of slides cropping ')
print('         {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
print('-------------------------')
print('')

print("----------------------------------")
print("|      Extracting tiles          |")
print("----------------------------------")

start = time.time()

##################################################
# TODO
##################################################

total_time = time.time() - start
print('-------------------------')
print('  End of tiles extracting ')
print('         {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
print('-------------------------')
print('')
