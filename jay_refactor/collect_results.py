from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import shutil
import time
from utils import DataFactory
from ThreeDiscrim import WGAN_Model
import game_visualizer
import matplotlib.pyplot as plt

os.environ[
    'TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'  # for mixed precision enable double lr and batch_size on nvidia tensorflow:19.03-py3

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'folder_path', None,
    '/workspace/data/nctu_cgvlab_bballgan/Log/.../results')
tf.app.flags.DEFINE_string(
    'check_point', None,
    '/workspace/data/nctu_cgvlab_bballgan/Log/.../Checkpoints/model.ckpt-176400'
)
tf.app.flags.DEFINE_string(
    'data_path', '/workspace/data/nctu_cgvlab_bballgan/Reordered_Data/',
    'summary directory')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size of input')
tf.app.flags.DEFINE_integer('latent_dims', 150, 'dimension of latent variable')
tf.app.flags.DEFINE_integer('seq_length', 50, 'sequence length')
tf.app.flags.DEFINE_integer('features_', 12, 'number of offence features')
tf.app.flags.DEFINE_integer('features_d', 10, 'number of defence features')
tf.app.flags.DEFINE_integer('n_resblock', 4, 'number of residual blocks')
tf.app.flags.DEFINE_integer('pretrain_D', 25, 'Epoch to pretrain D')
tf.app.flags.DEFINE_integer('train_D', 5, 'Number of times to train D')
tf.app.flags.DEFINE_float('lr_', 1e-4, 'learning rate')
tf.app.flags.DEFINE_float('lambda_', 1.0, 'Decaying lambda value')
tf.app.flags.DEFINE_integer('n_filters', 256, 'number of filters in conv')
tf.app.flags.DEFINE_float('keep_prob', 1.0, 'keep prob of dropout')

tf.app.flags.DEFINE_integer('checkpoint_step', 100,
                            'number of steps before saving checkpoint')


class Training_config(object):
    #Training configurations
    def __init__(self):
        self.folder_path = FLAGS.folder_path
        self.checkpoint_path = FLAGS.check_point
        self.data_path = FLAGS.data_path
        self.batch_size = FLAGS.batch_size
        self.latent_dims = FLAGS.latent_dims
        self.seq_length = FLAGS.seq_length
        self.features_ = FLAGS.features_
        self.features_d = FLAGS.features_d
        self.n_filters = FLAGS.n_filters
        self.lr_ = FLAGS.lr_
        self.keep_prob = FLAGS.keep_prob
        self.n_resblock = FLAGS.n_resblock

    def show(self):
        print(vars(self))


def z_samples(num_data):
    return np.random.normal(0., 1., size=[num_data, FLAGS.latent_dims])


class Collecter(object):
    def __init__(self, data_factory, config):
        self.data_factory = data_factory
        self.config = config
        self.model = WGAN_Model(config)
        self.num_data = self.data_factory.train_data['A'].shape[0]
        self.num_batch = self.num_data // config.batch_size
        self.num_batch_valid = self.data_factory.valid_data['A'].shape[
            0] // FLAGS.batch_size
        self.epoch_id = 0
        self.batch_id = 0
        self.batch_id_valid = 0
        print('self.num_batch:', self.num_batch)
        print('self.num_batch_valid:', self.num_batch_valid)
        self.model.load_model(config.checkpoint_path)

    def collect(self):
        num_valid_data = self.data_factory.valid_data['A'].shape[0]
        result = self.model.reconstruct_(self.data_factory.seq_valid,
                                         z_samples(num_valid_data),
                                         self.data_factory.f_valid)
        result = self.data_factory.recover_data(result[:, :, :22])
        np.save(
            os.path.join(self.config.folder_path, 'reconstruct.npy'), result)
        np.save(
            os.path.join(self.config.folder_path, 'seq_valid.npy'),
            self.data_factory.seq_valid)
        np.save(
            os.path.join(self.config.folder_path, 'feat_valid.npy'),
            self.data_factory.f_valid)

    def infer_diff_lengths(self):
        seq_data = np.load(os.path.join(FLAGS.data_path, 'Test2/TestSeq2.npy'))
        max_len = seq_data.shape[1]
        feat_data = np.load(os.path.join(FLAGS.data_path, 'Test2/TestSeqCond2.npy'))
        len_data = np.load(os.path.join(FLAGS.data_path, 'Test2/TestLength2.npy'))
        seq_data = self.data_factory.normalize(seq_data)
        num_data = seq_data.shape[0]
        results = []
        for i in range(num_data):
            print(i)
            res = self.model.reconstruct_(seq_data[i:i+1, :len_data[i]],
                                             z_samples(1),
                                             feat_data[i:i+1, :len_data[i]])
            res = np.concatenate([res, np.zeros(shape=[1,max_len-len_data[i],res.shape[2]])], axis=1)
            results.append(res)
        results = np.concatenate(results, axis=0)
        results = self.data_factory.recover_data(results[:, :, :22])
        np.save(
            os.path.join(self.config.folder_path, 'reconstruct.npy'), results)
        print(results.shape)


def main(_):
    with tf.get_default_graph().as_default() as graph:
        real_data = np.load(os.path.join(
            FLAGS.data_path, '50Real.npy'))[:, :FLAGS.seq_length, :, :]
        seq_data = np.load(os.path.join(FLAGS.data_path, '50Seq.npy'))
        features_ = np.load(os.path.join(FLAGS.data_path, 'SeqCond.npy'))
        real_feat = np.load(os.path.join(FLAGS.data_path, 'RealCond.npy'))

        print("Real Data: ", real_data.shape)
        print("Seq Data: ", seq_data.shape)
        print("Real Feat: ", real_feat.shape)
        print("Seq Feat: ", features_.shape)

        data_factory = DataFactory(
            real_data=real_data,
            seq_data=seq_data,
            features_=features_,
            real_feat=real_feat)

        config = Training_config()
        config.show()
        collector = Collecter(data_factory, config)
#         collector.collect()
        collector.infer_diff_lengths()


if __name__ == '__main__':
    if os.path.exists(FLAGS.folder_path):
        ans = input(
            '"%s" will be removed!! are you sure (y/N)? ' % FLAGS.folder_path)
        if ans == 'Y' or ans == 'y':
            # when not restore, remove follows (old) for new training
            shutil.rmtree(FLAGS.folder_path)
            print('rm -rf "%s" complete!' % FLAGS.folder_path)
        else:
            exit()
    if not os.path.exists(FLAGS.folder_path):
        os.makedirs(FLAGS.folder_path)
    assert FLAGS.check_point is not None
    tf.app.run()
