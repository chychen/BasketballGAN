from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from utils import DataFactory
from Train_Triple import Training_config
from ThreeDiscrim import WGAN_Model

DATA_PATH = os.path.join('./Data/')
MODEL_PATH = os.path.join(DATA_PATH,'Diff/model.ckpt-325')
SAVE_PATH = os.path.join(DATA_PATH,'vary_Length/')

BATCH_SIZE = 128
n_Latent =100
LATENT_DIM = 150
SEQ_LEN = 50

class Sampler():
    def __init__(self):
        self.data = None
        self.config = Training_config()
        self.model = WGAN_Model(self.config)
        self.model.load_model(MODEL_PATH)

def z_samples():
    return np.random.normal(0.,1.,size=[n_Latent,LATENT_DIM])

with tf.get_default_graph().as_default() as graph:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.import_meta_graph(MODEL_PATH + '.meta')
    print("Model found")
    with tf.Session(config=config) as sess:
        saver.restore(sess, MODEL_PATH)

        result_t = graph.get_tensor_by_name('G_/conv_result/conv1d/relu/Maximum:0')


        latent_input_t = graph.get_tensor_by_name('Latent:0')
        feature_input = graph.get_tensor_by_name('Seq_feat:0')
        condition_input_t = graph.get_tensor_by_name('Cond_input:0')
        #realfeat_input = graph.get_tensor_by_name('Real_feat:0')

        G_samples_t = graph.get_tensor_by_name('G_/conv_result/conv1d/relu/Maximum:0')
        matched_cond_t = graph.get_tensor_by_name('concat_3:0')
        # result tensor

        critic_scores_t = graph.get_tensor_by_name(
            'P_disc_1/conv_output/Reshape:0')

        critic_score_frames = graph.get_tensor_by_name(
            'P_disc_1/conv_output/conv1d/relu/Maximum:0')

        image_data = np.load('../Data/Model_data/50seq.npy')
        features_ = np.load('../Data/Model_data/50Cond.npy')
        real_data = np.load('../Data/Model_data/F50_D.npy')[:, :SEQ_LEN, :, :]
        real_feat = np.load('../Data/Model_data/RealCond.npy')
        print('real_data.shape', real_data.shape)

        data_factory = DataFactory(real_data, image_data, features_, real_feat)

        # target data
        target_data = np.load('../Data/Test/TestSeq.npy')[-600:-500]
        target_real_data = np.load('../Data/Test/TestReal.npy')[-600:-500]
        target_feat = np.load('../Data/Test/TestSeqCond.npy')[-600:-500]
        target_length = np.load('../Data/Test/TestLength.npy')[-600:-500]

        real_play = np.concatenate( [target_real_data[:,:,0,:2].reshape([target_data.shape[0],target_data.shape[1],1*2]),
                                    target_real_data[:,:,1:,:2].reshape([target_data.shape[0],target_data.shape[1], 10* 2])],
                                    axis = -1
                                    )
        seq_conditon = np.concatenate([   # ball
                target_data[:, :, :2].reshape(
                    [target_data.shape[0], target_data.shape[1], 1 * 2]),
                # team A players
                target_data[:, :,2:12].reshape(
                    [target_data.shape[0], target_data.shape[1], 5 * 2])
                ],axis=-1)

        team_AB = np.concatenate(
            [   # ball
                target_data[:, :, :2].reshape(
                    [target_data.shape[0], target_data.shape[1], 1 * 2]),
                # team A players
                target_data[:, :,2:12].reshape(
                    [target_data.shape[0], target_data.shape[1], 5 * 2]),
                #feature
                target_feat[:,:,:].reshape(
                    [target_feat.shape[0],target_feat.shape[1],6 * 1]
                )
            ], axis=-1
        )
        #team_AB = data_factory.normalize(team_AB)
        team_A = team_AB[:, :, :12]
        team_Feat = team_AB[:,:,12:]

        # result collector
        results_A_fake_B = []
        results_A_real_B = []
        results_critic_scores = []
        result_length = []
        plays = []
        fplays = []


        for idx in range(team_AB.shape[0]):
            # given 100(FLAGS.n_latents) latents generate 100 results on same condition at once
            real_conds = team_A[idx:idx + 1, :target_length[idx]]
            real_conds = np.concatenate(
                [real_conds for _ in range(n_Latent)], axis=0)

            real_feat = team_Feat[idx:idx + 1, :target_length[idx]]
            real_feat = np.concatenate(
                [real_feat for _ in range(n_Latent)], axis=0)

            real_cond_concat = np.concatenate([real_conds,real_feat],axis = -1)

            # generate result
            latents = z_samples()
            feed_dict = {
                latent_input_t: latents,
                condition_input_t: real_conds,
                feature_input: real_feat,
            }

            result = sess.run(
                result_t, feed_dict=feed_dict)

            # calculate em distance
            feed_dict = {
                G_samples_t: result,
                matched_cond_t: real_cond_concat
            }
            em_dist = sess.run(
                critic_scores_t, feed_dict=feed_dict)


            recoverd_A_fake_B = data_factory.recover_data(result[:,:,:22])

            recoverd_A_fake_B = np.concatenate([recoverd_A_fake_B,result[:,:,22:]],axis =-1)

            result_length.append(target_length[idx])

            # padding to length=200
            dummy = np.zeros(
                shape=[n_Latent, team_AB.shape[1] - target_length[idx], 28])

            temp_A_fake_B_concat = np.concatenate([recoverd_A_fake_B, dummy], axis= 1)

            results_A_fake_B.append(temp_A_fake_B_concat)

            real_plays = real_play[idx]
            fake_plays = seq_conditon[idx]


            plays.append(real_play)
            fplays.append(fake_plays)

            results_critic_scores.append(em_dist)

        print(np.array(results_A_fake_B).shape)
        print(np.array(results_critic_scores).shape)

        # concat along with conditions dimension (axis=1)
        results_A_fake_B = np.stack(results_A_fake_B, axis=1)
        results_critic_scores = np.stack(results_critic_scores, axis=1)

        result_length = np.stack(result_length,axis=0)
        plays = np.stack(plays,axis=0)
        fplays = np.stack(fplays, axis=0)
        # real data

        # saved as numpy
        print(np.array(results_A_fake_B).shape)
        print(np.array(results_A_real_B).shape)
        print(np.array(results_critic_scores).shape)

        #np.save(SAVE_PATH + 'result_Real_400v2.npy',
        #        plays)
        #np.save(SAVE_PATH + 'result_SeqCond.npy',
        #        fplays)

        np.save(SAVE_PATH+'result_length_600.npy',
                result_length)

        np.save(SAVE_PATH + 'results_Diff_v2.npy',
                np.array(results_A_fake_B).astype(np.float32).reshape([n_Latent, team_AB.shape[0], team_AB.shape[1], 28]))
        #np.save(SAVE_PATH + 'results_critic100_400_scores.npy',
        #        np.array(results_critic_scores).astype(np.float32).reshape([n_Latent, team_AB.shape[0]]))
        print('!!Completely Saved!!')
