import tensorflow as tf
import os
import numpy as np
from utils import DataFactory
import draw_feat

DATA_PATH = os.path.join('./Data/')
MODEL_PATH = os.path.join(DATA_PATH, 'checkpoints/model.ckpt-425')
SAVE_PATH = os.path.join(DATA_PATH,'output/')

BATCH_SIZE = 128
n_Latent = 100
LATENT_DIM = 150
SEQ_LEN = 50

def z_samples():
    return np.random.normal(0.,1.,size=[n_Latent,LATENT_DIM])

def Load_Model():

    with tf.get_default_graph().as_default() as graph:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.import_meta_graph(MODEL_PATH + '.meta')
        print("Model found")

        image_data = np.load('../Data/50seq.npy')
        features_ = np.load('../Data/50Cond.npy')
        real_data = np.load('../Data/F50_D.npy')[:, :50, :, :]
        real_feat = np.load('../Data/RealCond.npy')
        print('real_data.shape', real_data.shape)
        data_factory = DataFactory(real_data, image_data, features_, real_feat)

    return graph,saver,config,data_factory


def run_Model(graph,saver,config,data_factory):
    points = np.load('../Final/points2.npy')
    front = np.tile(points[0],(2,1))
    points = np.concatenate([front,points])
    print("Points:",points.shape)
    print(points[-1])
    extra = np.tile(points[-1],(2,1))
    #extra = np.reshape(extra,newshape=[5,12])
    print("Extra:",extra)
    points = np.concatenate([points,extra])

    feature = draw_feat.get_feature(points)

    with tf.Session(config=config) as sess:
        saver.restore(sess, MODEL_PATH)

        result_t = graph.get_tensor_by_name('G_/conv_result/conv1d/relu/add:0')
        '''
        latent_input_t = graph.get_tensor_by_name('Placeholder_7:0')
        feature_input = graph.get_tensor_by_name('Placeholder_3:0')
        condition_input_t = graph.get_tensor_by_name('Placeholder_6:0')
        '''

        latent_input_t = graph.get_tensor_by_name('Latent:0')
        feature_input = graph.get_tensor_by_name('Real_feat:0')
        condition_input_t = graph.get_tensor_by_name('Cond_input:0')

        print("Loaded")

        target_length = len(points)
        dims = 1
        points = [points] * dims
        points = np.reshape(points, newshape=[dims, target_length, 12])

        #feature = np.load('./Bhost.npy')

        feature = np.repeat(feature, dims, axis=0)

        print(points.shape)

        # target data
        target_data = points
        target_feat = feature

        print('target_data.shape', target_data.shape)

        team_AB = np.concatenate(
            [  # ball
                target_data[:, :, :2].reshape(
                    [target_data.shape[0], target_data.shape[1], 1 * 2]),
                # team A players
                target_data[:, :, 2:12].reshape(
                    [target_data.shape[0], target_data.shape[1], 5 * 2]),
                # feature
                target_feat[:, :, :].reshape(
                    [target_feat.shape[0], target_feat.shape[1], 6 * 1]
                )
            ], axis=-1
        )
        team_AB = data_factory.normalize(team_AB)
        team_A = team_AB[:, :, :12]
        team_Feat = team_AB[:, :, 12:]

        # result collector
        results_A_fake_B = []
        results_A_real_B = []

        print(team_AB.shape)
        print(team_AB.shape[0])

        for idx in range(team_AB.shape[0]):
            # given 100(FLAGS.n_latents) latents generate 100 results on same condition at once
            real_conds = team_A[idx:idx + 1, :target_length]
            real_conds = np.concatenate(
                [real_conds for _ in range(n_Latent)], axis=0)

            real_feat = team_Feat[idx:idx + 1, :target_length]
            real_feat = np.concatenate(
                [real_feat for _ in range(n_Latent)], axis=0)

            # generate result
            latents = z_samples()
            feed_dict = {
                latent_input_t: latents,
                condition_input_t: real_conds,
                feature_input: real_feat
            }

            result = sess.run(
                result_t, feed_dict=feed_dict)

            recoverd_A_fake_B = data_factory.recover_data(result[:, :, :22])

            recoverd_A_fake_B = np.concatenate([recoverd_A_fake_B, result[:, :, 22:]], axis=-1)

            temp_A_fake_B_concat = recoverd_A_fake_B

            results_A_fake_B.append(temp_A_fake_B_concat)

        # concat along with conditions dimension (axis=1)
        results_A_fake_B = np.stack(results_A_fake_B, axis=1)
        # real data

        # saved as numpy
        print(np.array(results_A_fake_B).shape)
        print(np.array(results_A_real_B).shape)

        np.save(SAVE_PATH + 'output.npy',
                np.array(results_A_fake_B).astype(np.float32).reshape(
                    [n_Latent, team_AB.shape[0], team_AB.shape[1], 28]))

        print('!!Completely Saved!!')
