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
    'folder_path', '/workspace/data/nctu_cgvlab_bballgan/Log/new_folder/',
    'summeray directory')
tf.app.flags.DEFINE_string('check_point', None, 'summary directory')
tf.app.flags.DEFINE_string(
    'data_path', '/workspace/data/nctu_cgvlab_bballgan/Reordered_Data/',
    'summary directory')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size of input')
tf.app.flags.DEFINE_integer('latent_dims', 150, 'dimension of latent variable')
tf.app.flags.DEFINE_integer('seq_length', 50, 'sequence length')
tf.app.flags.DEFINE_integer('features_', 12, 'number of offence features')
tf.app.flags.DEFINE_integer('features_d', 10, 'number of defence features')
tf.app.flags.DEFINE_integer('n_resblock', 8, 'number of residual blocks')
tf.app.flags.DEFINE_integer('pretrain_D', 25, 'Epoch to pretrain D')
tf.app.flags.DEFINE_integer('train_D', 5, 'Number of times to train D')
tf.app.flags.DEFINE_float('lr_', 1e-4, 'learning rate')
tf.app.flags.DEFINE_float('lambda_', 1.0, 'Decaying lambda value')
tf.app.flags.DEFINE_integer('n_filters', 256, 'number of filters in conv')
tf.app.flags.DEFINE_float('keep_prob', 1.0, 'keep prob of dropout')

tf.app.flags.DEFINE_integer('checkpoint_step', 100,
                            'number of steps before saving checkpoint')

CHECKPOINT_PATH = os.path.join(FLAGS.folder_path, 'Checkpoints/')
SAMPLE_PATH = os.path.join(FLAGS.folder_path, 'Samples/')


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


#Generate sample
def encode(data, model):
    new_shape = list(data.shape)
    return model.gen_latent(np.reshape(data, new_shape))


def reconstruct_(model, x, z, x2):
    return model.reconstruct_(x, z, x2)


def z_samples():
    return np.random.normal(0., 1., size=[FLAGS.batch_size, FLAGS.latent_dims])


class Trainer(object):
    def __init__(self, data_factory, config):
        self.data_factory = data_factory
        self.config = config
        self.model = WGAN_Model(config)
        self.num_data = self.data_factory.train_data['A'].shape[0]
        self.num_batch = self.num_data // FLAGS.batch_size
        self.num_batch_valid = self.data_factory.valid_data['A'].shape[
            0] // FLAGS.batch_size
        self.epoch_id = 0
        self.batch_id = 0
        self.batch_id_valid = 0
        print('self.num_batch:', self.num_batch)
        print('self.num_batch_valid:', self.num_batch_valid)

    def __call__(self):
        while True:
            if self.epoch_id < FLAGS.pretrain_D == 0:  # warming
                num_d = 10
            else:
                num_d = FLAGS.train_D
            start_time = time.time()
            for _ in range(num_d):
                self.train_D()


#             print('D img/s:{}'.format(num_d*FLAGS.batch_size/(time.time()-start_time)))
            start_time = time.time()
            self.train_G()
            #             print('G img/s:{}'.format(FLAGS.batch_size/(time.time()-start_time)))
            # validation
            valid_idx = self.batch_id_valid * FLAGS.batch_size
            valid_ = self.data_factory.valid_data['A'][valid_idx:valid_idx +
                                                       FLAGS.batch_size]
            valid_D = self.data_factory.valid_data['B'][valid_idx:valid_idx +
                                                        FLAGS.batch_size]

            valid_ = valid_[:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
            seq_v = self.data_factory.seq_valid[valid_idx:valid_idx +
                                                FLAGS.batch_size]
            rv_feat = self.data_factory.f_valid[valid_idx:valid_idx +
                                                FLAGS.batch_size]
            rfv_feat = self.data_factory.rf_valid[valid_idx:valid_idx +
                                                  FLAGS.batch_size]

            self.model.valid_loss(
                x=valid_,
                x2=valid_D,
                y=seq_v,
                z=z_samples(),
                feat_=rv_feat,
                feat2_=rfv_feat)
            self.update_batch_id_valid_and_shuffle()

    def train_G(self):
        data_idx = self.batch_id * FLAGS.batch_size
        training_data = self.data_factory.train_data
        f_train = self.data_factory.f_train
        seq_train = self.data_factory.seq_train
        rf_train = self.data_factory.rf_train

        real_ = training_data['A'][data_idx:data_idx + FLAGS.batch_size]
        real_D = training_data['B'][data_idx:data_idx + FLAGS.batch_size]

        seq_feat = f_train[data_idx:data_idx + FLAGS.batch_size]
        real_ = real_[:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        seq_ = seq_train[data_idx:data_idx + FLAGS.batch_size]

        real_feat = rf_train[data_idx:data_idx + FLAGS.batch_size]

        self.model.update_gen(
            real=real_,
            real_d=real_D,
            x=seq_,
            x2=seq_feat,
            x3=real_feat,
            z=z_samples())

        self.update_batch_id_and_shuffle()

    def train_D(self):
        data_idx = self.batch_id * FLAGS.batch_size
        training_data = self.data_factory.train_data
        f_train = self.data_factory.f_train
        seq_train = self.data_factory.seq_train
        rf_train = self.data_factory.rf_train

        real_ = training_data['A'][data_idx:data_idx + FLAGS.batch_size]
        real_D = training_data['B'][data_idx:data_idx + FLAGS.batch_size]

        seq_feat = f_train[data_idx:data_idx + FLAGS.batch_size]
        real_ = real_[:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        seq_ = seq_train[data_idx:data_idx + FLAGS.batch_size]

        real_feat = rf_train[data_idx:data_idx + FLAGS.batch_size]

        self.model.update_discrim(
            x=real_,
            x2=real_D,
            y=seq_,
            z=z_samples(),
            feat_=seq_feat,
            feat2_=real_feat)

        self.update_batch_id_and_shuffle()

    def update_batch_id_valid_and_shuffle(self):
        self.batch_id_valid = self.batch_id_valid + 1
        if self.batch_id_valid >= self.num_batch_valid:
            self.batch_id_valid = 0
            self.data_factory.shuffle_valid()

    def update_batch_id_and_shuffle(self):
        self.batch_id = self.batch_id + 1
        if self.batch_id >= self.num_batch:
            self.epoch_id = self.epoch_id + 1
            self.batch_id = 0
            self.data_factory.shuffle_train()
            # save model
            if self.epoch_id % FLAGS.checkpoint_step == 0:
                checkpoint_ = os.path.join(CHECKPOINT_PATH, 'model.ckpt')
                self.model.save_model(checkpoint_)
                print("Saved model:", checkpoint_)
            # save generated sample
            if self.epoch_id % 10 == 0:
                print('epoch_id:', self.epoch_id)
                data_idx = self.batch_id * FLAGS.batch_size
                f_train = self.data_factory.f_train
                seq_train = self.data_factory.seq_train
                seq_feat = f_train[data_idx:data_idx + FLAGS.batch_size]
                seq_ = seq_train[data_idx:data_idx + FLAGS.batch_size]

                recon = reconstruct_(self.model, seq_, z_samples(), seq_feat)
                sample = recon[:, :, :22]
                samples = self.data_factory.recover_BALL_and_A(sample)
                samples = self.data_factory.recover_B(samples)
                game_visualizer.plot_data(
                    samples[0],
                    FLAGS.seq_length,
                    file_path=SAMPLE_PATH + 'reconstruct{}.mp4'.format(
                        self.epoch_id),
                    if_save=True)


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
        trainer = Trainer(data_factory, config)
        trainer()


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
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    if not os.path.exists(SAMPLE_PATH):
        os.makedirs(SAMPLE_PATH)
    tf.app.run()
