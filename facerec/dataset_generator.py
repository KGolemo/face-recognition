import os
import sys
import csv
import math
import numpy as np
import tensorflow as tf
from typing import List, Tuple


class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for Siamese network training.

    Parameters
    ----------
    positive_pairs_path : str
        Path to the CSV file containing positive pairs.
    negative_pairs_path : str
        Path to the CSV file containing negative pairs.
    images_path : str
        Path to the folder containing image data.
    input_shape : Tuple[int, int, int]
        The desired shape for input images (height, width, channels).
    batch_size : int
        Batch size for data generation.
    seed : int
        Random seed for shuffling data.
    shuffle : bool
        Whether to shuffle the data at the end of each epoch.
    debug : bool, optional
        Whether to enable debug mode. Defaults to False.
    """

    def __init__(self, positive_pairs_path: str, negative_pairs_path: str, images_path: str,
                 input_shape: Tuple[int, int, int], batch_size: int, seed: int, shuffle: bool, debug: bool = False):
        self.pos_pairs_path = positive_pairs_path
        self.neg_pairs_path = negative_pairs_path
        self.imgs_path = images_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.debug = debug

        if not os.path.isfile(positive_pairs_path):
            print("File path {} does not exist. Exiting...".format(positive_pairs_path))
            sys.exit()

        if not os.path.isfile(negative_pairs_path):
            print("File path {} does not exist. Exiting...".format(negative_pairs_path))
            sys.exit()

        if not os.path.isdir(images_path):
            print("Images folder {} does not exist. Exiting...".format(images_path))
            sys.exit()

        left_pos_imgs_paths, right_pos_imgs_paths = self.__get_images_paths(self.pos_pairs_path, self.imgs_path)
        left_neg_imgs_paths, right_neg_imgs_paths = self.__get_images_paths(self.neg_pairs_path, self.imgs_path)

        pos_imgs_labels = [1] * len(left_pos_imgs_paths)
        neg_imgs_labels = [0] * len(left_neg_imgs_paths)

        self.left_imgs_paths = left_pos_imgs_paths + left_neg_imgs_paths
        self.right_imgs_paths = right_pos_imgs_paths + right_neg_imgs_paths
        self.labels = pos_imgs_labels + neg_imgs_labels
        self.indices = range(0, len(self.left_imgs_paths))
        self.on_epoch_end()

    @staticmethod
    def __get_images_paths(csv_path: str, imgs_path: str) -> Tuple[List[str], List[str]]:
        """
        Retrieve image paths from a CSV file.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing pairs.
        imgs_path : str
            Path to the folder containing images.

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple containing lists of left and right image paths.
        """

        left_imgs_paths = []
        right_imgs_paths = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for items in reader:
                if len(items) == 2:
                    name = items[0]
                    pair = eval(items[1])
                    left_img_src = os.path.join(imgs_path, name, pair[0])
                    right_img_src = os.path.join(imgs_path, name, pair[1])
                elif len(items) == 3:
                    name1 = items[0]
                    name2 = items[1]
                    pair = eval(items[2])
                    left_img_src = os.path.join(imgs_path, name1, pair[0])
                    right_img_src = os.path.join(imgs_path, name2, pair[1])
                left_imgs_paths.append(left_img_src)
                right_imgs_paths.append(right_img_src)
        return left_imgs_paths, right_imgs_paths

    def __len__(self) -> int:
        """
        Get the number of batches per epoch.

        Returns
        -------
        int
            The number of batches per epoch.
        """
        return math.ceil(len(self.indices) / self.batch_size)

    def __center_crop(self, img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """
        Perform center cropping on an image.

        Parameters
        ----------
        img : numpy.ndarray
            Input image.
        new_shape : tuple[int, int]
            The desired shape for the cropped image (height, width).

        Returns
        -------
        numpy.ndarray
            Cropped image.
        """
        x, y, _ = img.shape
        start_x = x//2 - new_shape[0]//2
        start_y = y//2 - new_shape[1]//2
        return img[start_x:start_x+new_shape[1], start_y:start_y+new_shape[0], :]

    def __getitem__(self, index: int) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Generate a batch of data.

        Parameters
        ----------
        index : int
            Index of the batch.

        Returns
        -------
        tuple[tuple[np.ndarray, np.ndarray], np.ndarray]
            A tuple containing the left and right input image batches and the label batch.
        """
        low = index * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.left_imgs_paths))

        batch_index = self.index[low:high]
        X_left_batch = [self.left_imgs_paths[k] for k in batch_index]
        X_right_batch = [self.right_imgs_paths[k] for k in batch_index]
        y_batch = [self.labels[k] for k in batch_index]

        X_batch = self.__get_data(X_left_batch, X_right_batch)
        return X_batch, np.array(y_batch)

    def __get_data(self, left_imgs_paths: List[str], right_imgs_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess image data.

        Parameters
        ----------
        left_imgs_paths : list[str]
            List of paths to left images.
        right_imgs_paths : list[str]
            List of paths to right images.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing left and right image batches.
        """
        left_imgs_batch = []
        for l_path in left_imgs_paths:
            left_imgs_batch.append(self.__get_image(l_path))

        right_imgs_batch = []
        for r_path in right_imgs_paths:
            right_imgs_batch.append(self.__get_image(r_path))

        return np.asarray(left_imgs_batch), np.asarray(right_imgs_batch)

    def __get_image(self, img_path: str, center_crop: bool = True) -> np.ndarray:
        """
        Load and preprocess a single image.

        Parameters
        ----------
        img_path : str
            Path to the image file.
        center_crop : bool, optional
            Whether to perform center cropping. Defaults to True.

        Returns
        -------
        numpy.ndarray
            Processed image data.
        """
        img = tf.keras.utils.load_img(img_path)
        img_arr = tf.keras.utils.img_to_array(img)

        if center_crop:
            img_arr = self.__center_crop(img_arr, (self.input_shape[0], self.input_shape[1]))
        else:
            img_arr = tf.image.resize(img_arr, (self.input_shape[0], self.input_shape[1])).numpy()

        return img_arr/255.

    def on_epoch_end(self):
        """
        Shuffle data indices at the end of each epoch.
        """
        self.index = np.arange(len(self.indices))
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.index)


def train_val_test_split(pairs_path: str, train_size: float, val_size: float, test_size: float, seed: int):
    """
    Split data from a pairs file into training, validation, and test sets and save them to separate files.

    Parameters
    ----------
    pairs_path : str
        Path to the pairs file containing the data.
    train_size : float
        Proportion of data to include in the training set (0.0 to 1.0).
    val_size : float
        Proportion of data to include in the validation set (0.0 to 1.0).
    test_size : float
        Proportion of data to include in the test set (0.0 to 1.0).
    seed : int
        Random seed for shuffling data.

    Raises
    ------
    AssertionError
        If the sum of `train_size`, `val_size`, and `test_size` is not equal to 1.

    Note
    ----
    The `train_size`, `val_size`, and `test_size` should sum up to 1.

    Example
    -------
    >>> train_val_test_split("pairs.txt", 0.7, 0.2, 0.1, 42)
    """
    pairs = open(pairs_path, 'r')
    lines = pairs.readlines()
    pos_header = lines.pop(0)
    pairs.close()

    assert train_size + val_size + test_size == 1.0

    np.random.seed(seed)
    np.random.shuffle(lines)

    path, filename = os.path.split(pairs_path)

    train_split = int(train_size * len(lines))
    val_split = int(val_size * len(lines))
    test_split = int(test_size * len(lines))

    pairs_shuffled = open(os.path.join(path, 'train_'+filename), 'w')
    pairs_shuffled.writelines(pos_header)
    pairs_shuffled.writelines(lines[:train_split])
    pairs_shuffled.close()

    pairs_shuffled = open(os.path.join(path, 'val_'+filename), 'w')
    pairs_shuffled.writelines(pos_header)
    pairs_shuffled.writelines(lines[train_split:train_split+val_split])
    pairs_shuffled.close()

    pairs_shuffled = open(os.path.join(path, 'test_'+filename), 'w')
    pairs_shuffled.writelines(pos_header)
    pairs_shuffled.writelines(lines[train_split+val_split:train_split+val_split+test_split])
    pairs_shuffled.close()
