import tensorflow as tf
import numpy as np
import json


class DataGenerator(tf.keras.utils.Sequence):

    # define file list of X and y the model needs to go through in an epoch
    def __init__(self, file_list_x, file_list_y):
        self.file_list_X = file_list_x
        self.file_list_y = file_list_y
        self.indexes = np.arange(len(self.file_list_X))
        with open('stats/data_stats.json') as f:
            self.data_stats = json.load(f)

    # length is the number of batches
    def __len__(self):
        return int(len(self.file_list_X))

    # get items using indexes
    def __getitem__(self, index):
        indexes = self.indexes[index:(index + 1)]

        file_list_temp_X = [self.file_list_X[idx] for idx in indexes]
        file_list_temp_y = [self.file_list_y[idx] for idx in indexes]

        X, y = self.__data_generation(file_list_temp_X, file_list_temp_y)

        return (X, y)

    # get X and y batch
    def __data_generation(self, file_list_temp_X, file_list_temp_y):

        for X_path, y_path in zip(file_list_temp_X, file_list_temp_y):
            X_batch = np.load(X_path)
            y_batch = np.load(y_path)

        # normalize/standardize here
        X_batch = self.standardization(X_batch,
                                       mean=self.data_stats['data_mean'],
                                       std=self.data_stats['data_std'])

        return (X_batch, y_batch)

    # linear normalization
    def linear_norm(self, x, x_min, x_max):
        x_norm = (x - x_min) / (x_max - x_min)
        return x_norm

    # standardization
    def standardization(self, x, mean, std):
        # also known as Z score, moves mean to zero and std to one
        x_norm = (x - mean) / std
        return x_norm

