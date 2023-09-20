from itertools import combinations
import numpy as np
import csv
import sys
import os


class PairsGenerator:
    """
    Generates positive and negative pairs of images for training a Siamese network.

    Parameters
    ----------
    images_directory : str
        The path to the directory containing the images.

    Attributes
    ----------
    images_directory : str
        The path to the directory containing the images.
    pairs_directory : str
        The path to the directory where generated pairs will be stored.
    positive_pairs : list[list[str, list[str]]]
        A list to store positive pairs of images.
    negative_pairs : list[list[str, str, list[str]]]
        A list to store negative pairs of images.
    """

    def __init__(self, images_directory: str):
        """
        Initializes the PairsGenerator with the path to the images directory.

        Parameters
        ----------
        images_directory : str
            The path to the directory containing the images.

        Raises
        ------
        SystemExit
            If the images_directory does not exist.
        """
        if not os.path.isdir(images_directory):
            print("Images folder {} does not exist. Exiting...".format(images_directory))
            sys.exit()
        self.images_directory = images_directory

        pairs_directory = os.path.join(os.getcwd(), 'data', 'pairs')
        if not os.path.isdir(pairs_directory):
            os.mkdir(pairs_directory)
        self.pairs_directory = pairs_directory

        self.positive_pairs = []
        self.negative_pairs = []

    def generate_positive_pairs(self):
        """
        Generates positive pairs of images.

        This method generates positive pairs by selecting combinations of two images
        from the same person's directory.

        Notes
        -----
        The maximum number of positive pairs generated is set to 49 per person.
        """
        for person_name in os.listdir(self.images_directory):
            person_images = os.listdir(os.path.join(self.images_directory, person_name))
            person_pairs = list(combinations(person_images, 2))[:49]
            for pair in person_pairs:
                self.positive_pairs.append([person_name, list(pair)])

    def export_positive_pairs_to_csv(self, filename: str = 'positive_pairs.csv'):
        """
        Exports positive pairs to a CSV file.

        Parameters
        ----------
        filename : str, optional
            The name of the CSV file to export positive pairs to (default is 'positive_pairs.csv').
        """
        if not os.path.exists(os.path.join(self.pairs_directory, filename)):
            with open(os.path.join(self.pairs_directory, filename), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'positive_pairs'])
                writer.writerows(self.positive_pairs)

    def get_all_images_names(self):
        """
        Gets the names of all images in the directory.

        Returns
        -------
        list[str]
            A list of image names.
        """
        imgs_names = []
        for person_name in os.listdir(self.images_directory):
            person_images = os.listdir(os.path.join(self.images_directory, person_name))
            imgs_names.extend(person_images)
        return imgs_names

    def generate_negative_combinations(self, combinations_num: int = 15080):
        """
        Generates negative pairs of images.

        This method generates negative pairs by selecting random combinations of two images
        from different people's directories.

        Parameters
        ----------
        combinations_num : int, optional
            The number of negative combinations to generate (default is 15080).

        Notes
        -----
        This method uses random sampling to create negative pairs.
        """
        all_imgs = self.get_all_images_names()
        used_pairs = set()

        while len(self.negative_pairs) < combinations_num:
            for img1 in list(all_imgs):
                name1 = img1[:-9]

                available_imgs = list(filter(lambda x: not x.startswith(name1), all_imgs))

                img2 = np.random.choice(available_imgs)
                pair = (img1, img2)
                while tuple(sorted(pair)) in used_pairs:
                    img2 = np.random.choice(available_imgs)
                    pair = (img1, img2)
                name2 = img2[:-9]
                used_pairs.add(tuple(sorted(pair)))
                self.negative_pairs.append([name1, name2, list(pair)])
                if len(self.negative_pairs) == combinations_num:
                    break

    def export_negative_pairs_to_csv(self, filename: str = 'negative_pairs.csv'):
        """
        Exports negative pairs to a CSV file.

        Parameters
        ----------
        filename : str, optional
            The name of the CSV file to export negative pairs to (default is 'negative_pairs.csv').
        """
        if not os.path.exists(os.path.join(self.pairs_directory, filename)):
            with open(os.path.join(self.pairs_directory, filename), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['name1', 'name2', 'negative_pairs'])
                writer.writerows(self.negative_pairs)


if __name__ == '__main__':
    IMGS_DIR = os.path.join(os.getcwd(), 'data', 'faces')
    gen = PairsGenerator(images_directory=IMGS_DIR)
    gen.generate_negative_combinations()
    gen.export_negative_pairs_to_csv()
