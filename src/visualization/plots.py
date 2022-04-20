import matplotlib.pyplot as plt
import numpy as np


def plot_np_img(np_img: np.ndarray,
                color_map: str = 'gray',
                save_path: str = None,
                verbose: bool = False
                ) -> None:
    """
    Plot a numpy image

    :param np_img: grayscale image as numpy image
    :param color_map:
    :param save_path: path to save the plot
    :param verbose
    :return: None
    """
    if verbose: print("Plotting image from numpy image.")
    plt.figure()
    plt.imshow(np_img, cmap=color_map)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.clf()
    plt.cla()
    plt.close()
