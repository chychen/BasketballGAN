from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import os


class DataFactory(object):
    """singletance pattern
    """
    __instance = None

    def __new__(clz,
                real_data=None,
                seq_data=None,
                features_=None,
                real_feat=None):
        if not DataFactory.__instance:
            DataFactory.__instance = object.__new__(clz)
        else:
            print("Instance Exists! :D")
        return DataFactory.__instance

    def __init__(self,
                 real_data=None,
                 seq_data=None,
                 features_=None,
                 real_feat=None):
        """
        params
        ------
        real_data : float, shape=[#, length=100, players=11, features=4]
        note
        ----
        feature :
            x, y, z, and player position
        """
        if real_data is not None:
            self.__real_data = real_data
            self.__seq_data = seq_data
            self.features_ = features_
            self.real_feat = real_feat

            self.BASKET_LEFT = [4, 25]
            self.BASKET_RIGHT = [90, 25]
            # position normalization
            self.__norm_dict = self.__normalize_pos()
            # make training data ready
            self.train_data, self.valid_data,\
            self.seq_train,self.seq_valid,\
            self.f_train,self.f_valid,\
            self.rf_train,self.rf_valid = self.__get_ready()

    def fetch_ori_data(self):
        return np.concatenate(
            [
                # ball
                self.__real_data[:, :, 0, :3].reshape([
                    self.__real_data.shape[0], self.__real_data.shape[1], 1 * 3
                ]),
                # team A players
                self.__real_data[:, :, 1:, :2].reshape([
                    self.__real_data.shape[0], self.__real_data.shape[1],
                    10 * 2
                ])
            ],
            axis=-1)

    def recover_data(self, norm_data):
        # X
        norm_data[:, :, [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
        ]] = norm_data[:, :, [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
        ]] * self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean']
        # Y
        norm_data[:, :, [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
        ]] = norm_data[:, :, [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
        ]] * self.__norm_dict['y']['stddev'] + self.__norm_dict['y']['mean']
        return norm_data

    def recover_play(self, norm_data):
        # X
        norm_data[:, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]] = norm_data[:, [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
        ]] * self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean']
        # Y
        norm_data[:, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] = norm_data[:, [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
        ]] * self.__norm_dict['y']['stddev'] + self.__norm_dict['y']['mean']
        return norm_data

    def recover_seq(self, norm_data):
        # X
        norm_data[:, :, [0, 2, 4, 6, 8, 10]] = norm_data[:,:, [0, 2, 4, 6, 8, 10]] * \
            self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean']
        #norm_data[:, [0, 2, 4, 6, 8]] = norm_data[:, [0, 2, 4, 6, 8]] * \
        #                                    self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean']
        # Y
        norm_data[:,:, [1, 3, 5, 7, 9, 11]] = norm_data[:, :, [1, 3, 5, 7, 9,11]] * \
            self.__norm_dict['y']['stddev'] + self.__norm_dict['y']['mean']
        # Z

        return norm_data

    def recover_BALL_and_A(self, norm_data):
        # X
        norm_data[:, :, [0, 2, 4, 6, 8, 10]] = norm_data[ :,:, [0, 2, 4, 6, 8, 10]] * \
            self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean']
        #norm_data[:, [0, 2, 4, 6, 8]] = norm_data[:, [0, 2, 4, 6, 8]] * \
        #                                    self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean']
        # Y
        norm_data[ :, :, [1, 3, 5, 7, 9, 11]] = norm_data[ :,:, [1, 3, 5, 7, 9,11]] * \
            self.__norm_dict['y']['stddev'] + self.__norm_dict['y']['mean']
        # Z

        return norm_data

    def recover_B(self, norm_data):
        # X
        norm_data[:,:, [12,14, 16, 18, 20]] = norm_data[:,:,  [12, 14, 16,18, 20]] * \
            self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean']
        # Y
        norm_data[:,:,  [13, 15, 17, 19, 21]] = norm_data[:,:, [13, 15, 17, 19, 21]] * \
            self.__norm_dict['y']['stddev'] + self.__norm_dict['y']['mean']
        return norm_data

    def shuffle_train(self):
        shuffled_indexes = np.random.permutation(self.train_data['A'].shape[0])
        self.train_data['A'] = self.train_data['A'][shuffled_indexes]
        self.train_data['B'] = self.train_data['B'][shuffled_indexes]
        self.seq_train = self.seq_train[shuffled_indexes]
        self.f_train = self.f_train[shuffled_indexes]
        self.rf_train = self.rf_train[shuffled_indexes]

    def shuffle_valid(self):
        shuffled_indexes = np.random.permutation(self.valid_data['A'].shape[0])
        self.valid_data['A'] = self.valid_data['A'][shuffled_indexes]
        self.valid_data['B'] = self.valid_data['B'][shuffled_indexes]
        self.seq_valid = self.seq_valid[shuffled_indexes]
        self.f_valid = self.f_valid[shuffled_indexes]
        self.rf_valid = self.rf_valid[shuffled_indexes]

    def __get_ready(self):
        train = {}
        valid = {}
        # A
        team_A = np.concatenate(
            [  # ball
                self.__real_data[:, :, 0, :3].reshape([
                    self.__real_data.shape[0], self.__real_data.shape[1], 1 * 3
                ]),
                # team A players
                self.__real_data[:, :, 1:6, :2].reshape([
                    self.__real_data.shape[0], self.__real_data.shape[1], 5 * 2
                ])
            ],
            axis=-1)
        train['A'], valid['A'] = np.split(
            team_A, [self.__real_data.shape[0] // 10 * 9])

        s_train, s_valid = np.split(self.__seq_data,
                                    [self.__real_data.shape[0] // 10 * 9])
        # B
        team_B = self.__real_data[:, :, 6:11, :2].reshape(
            [self.__real_data.shape[0], self.__real_data.shape[1], 5 * 2])
        train['B'], valid['B'] = np.split(
            team_B, [self.__real_data.shape[0] // 10 * 9])

        f_train, f_valid = np.split(self.features_,
                                    [self.features_.shape[0] // 10 * 9])

        rf_train, rf_valid = np.split(self.real_feat,
                                      [self.real_feat.shape[0] // 10 * 9])

        return train, valid, s_train, s_valid, f_train, f_valid, rf_train, rf_valid

    def __normalize_pos(self):
        """ directly normalize player x,y,z on self.__real_data
        """
        norm_dict = {}
        axis_list = ['x', 'y', 'z']

        for i, axis_ in enumerate(axis_list):
            if axis_ == 'z':  # z
                mean_ = np.mean(self.__real_data[:, :, 0, i])
                stddev_ = np.std(self.__real_data[:, :, 0, i])
                self.__real_data[:, :, 0, i] = (self.__real_data[:, :, 0, i] -
                                                mean_) / stddev_
                norm_dict[axis_] = {}
                norm_dict[axis_]['mean'] = mean_
                norm_dict[axis_]['stddev'] = stddev_
            else:  # x and y
                mean_ = np.mean(self.__real_data[:, :, :, i])
                stddev_ = np.std(self.__real_data[:, :, :, i])
                self.__real_data[:, :, :, i] = (self.__real_data[:, :, :, i] -
                                                mean_) / stddev_

                self.BASKET_LEFT[i] = (self.BASKET_LEFT[i] - mean_) / stddev_
                self.BASKET_RIGHT[i] = (self.BASKET_RIGHT[i] - mean_) / stddev_
                norm_dict[axis_] = {}
                norm_dict[axis_]['mean'] = mean_
                norm_dict[axis_]['stddev'] = stddev_
        return norm_dict

    def normalize(self, input_):
        """ normalize player x,y,z on input
        input_ : shape=[128, 100, 23]
        """
        # x
        input_[:, :, [0, 2, 4, 6, 8, 10]] = ( input_[:, :,[0, 2, 4, 6, 8, 10]] -
                                                                    self.__norm_dict['x']['mean']) / \
                                                                self.__norm_dict['x']['stddev']
        # y
        input_[:, :, [1, 3, 5, 7, 9, 11]] = (input_[:, :, [1, 3, 5, 7, 9, 11]] -
                                                                     self.__norm_dict['y']['mean']) / \
                                                                 self.__norm_dict['y']['stddev']

        return input_


def testing_real():
    pass


if __name__ == '__main__':
    testing_real()
