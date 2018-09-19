########################################################################
#
# Functions for downloading the CIFAR-10 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import pickle
import os
import download
from dataset import one_hot_encoded

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "data/CIFAR-10/"

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

class CIFAR10:
    

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
    img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
    num_channels = 3

# Length of an image when flattened to a 1-dim array.
    img_size_flat = img_size * img_size * num_channels

# Number of classes.
    num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
    _num_files_train = 5

# Number of images for each batch-file in the training-set.
    _images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
    _num_images_train = _num_files_train * _images_per_file

########################################################################
# Private functions for downloading, unpacking and loading data-files.
    def __init__(self, data_dir="data/CIFAR-10/"):
 

        # Copy args to self.
        self.data_dir = data_dir
        
        # Number of images in each sub-set.
#        self.num_train = _num_images_train
#        self.num_val = 5000
#        self.num_test = 10000

        # Download / load the training-set.
        self.maybe_download_and_extract()
        self.name=self.load_class_names()
        self.x_train,self.y_train_cls,self.y_train = self.load_training_data()
        self.x_test,self.y_test_cls,self.y_test = self.load_test_data()



    def _get_file_path(self,filename=""):
        """
        Return the full path of a data-file for the data-set.

        If filename=="" then return the directory of the files.
        """

        return os.path.join(self.data_dir, "cifar-10-batches-py/", filename)


    def _unpickle(self,filename):
        """
        Unpickle the given file and return the data.

        Note that the appropriate dir-name is prepended the filename.
        """

    # Create full path for the file.
        file_path = self._get_file_path(filename)

        print("Loading data: " + file_path)

        with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
            data = pickle.load(file, encoding='bytes')

        return data


    def _convert_images(self,raw):
        """
        Convert images from the CIFAR-10 format and
         a 4-dim array with shape: [image_number, height, width, channel]
         where the pixels are floats between 0.0 and 1.0.
         """

    # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, self.num_channels, self.img_size, self.img_size])

    # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])

        return images


    def _load_data(self,filename):
        """
        Load a pickled data-file from the CIFAR-10 data-set
        and return the converted images (see above) and the class-number
        for each image.
        """

    # Load the pickled data-file.
        data = self._unpickle(filename)

    # Get the raw images.
        raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
        cls = np.array(data[b'labels'])

    # Convert the images.
        images = self._convert_images(raw_images)

        return images, cls


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


    def maybe_download_and_extract(self):
        """
        Download and extract the CIFAR-10 data-set if it doesn't already exist
        in data_path (set this variable first to the desired path).
        """

        download.maybe_download_and_extract(url=data_url, download_dir=data_path)


    def load_class_names(self):
        """
        Load the names for the classes in the CIFAR-10 data-set.

        Returns a list with the names. Example: names[3] is the name
        associated with class-number 3.
        """

    # Load the class-names from the pickled file.
        raw = self._unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
        names = [x.decode('utf-8') for x in raw]

        return names


    def load_training_data(self):
        """
        Load all the training-data for the CIFAR-10 data-set.

        The data-set is split into 5 data-files which are merged here.

        Returns the images, class-numbers and one-hot encoded class-labels.
        """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
        images = np.zeros(shape=[self._num_images_train, self.img_size, self.img_size, self.num_channels], dtype=float)
        cls = np.zeros(shape=[self._num_images_train], dtype=int)

    # Begin-index for the current batch.
        begin = 0

    # For each data-file.
        for i in range(self._num_files_train):
        # Load the images and class-numbers from the data-file.
            images_batch, cls_batch = self._load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
            num_images = len(images_batch)

        # End-index for the current batch.
            end = begin + num_images

        # Store the images into the array.
            images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
            cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
            begin = end

        return images, cls, one_hot_encoded(class_numbers=cls, num_classes=self.num_classes)


    def load_test_data(self):
        """
        Load all the test-data for the CIFAR-10 data-set.

        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        images, cls = self._load_data(filename="test_batch")

        return images, cls, one_hot_encoded(class_numbers=cls, num_classes=self.num_classes)
    
    def random_batch(self, batch_size=32):
        """
        Create a random batch of training-data.

        :param batch_size: Number of images in the batch.
        :return: 3 numpy arrays (x, y, y_cls)
        """

        # Create a random index into the training-set.
        idx = np.random.randint(low=0, high=self._num_images_train, size=batch_size)

        # Use the index to lookup random training-data.
        x_batch = self.x_train[idx]
        y_batch = self.y_train[idx]
        y_batch_cls = self.y_train_cls[idx]

        return x_batch, y_batch, y_batch_cls

########################################################################
