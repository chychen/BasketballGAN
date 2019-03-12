from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
from utils import DataFactory
from ThreeDiscrim import WGAN_Model
import game_visualizer
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('folder_path','./Data/',
                           'summeray directory')
tf.app.flags.DEFINE_string('check_point',None,
                           'summary directory')
tf.app.flags.DEFINE_string('log_dir','./Log',
                           'Log summary tensorboard')
tf.app.flags.DEFINE_string('data_path','../Data/',
                           'summary directory')
tf.app.flags.DEFINE_integer('batch_size',128,
                            'batch size of input')
tf.app.flags.DEFINE_integer('latent_dims',150,
                            'dimension of latent variable')
tf.app.flags.DEFINE_integer('seq_length',50,
                            'sequence length')
tf.app.flags.DEFINE_integer('features_',12,
                            'number of offence features')
tf.app.flags.DEFINE_integer('features_d',10,
                            'number of defence features')
tf.app.flags.DEFINE_integer('n_resblock',8,
                            'number of residual blocks')
tf.app.flags.DEFINE_integer('pretrain_D',100,
                            'Epoch to pretrain D')
tf.app.flags.DEFINE_integer('freq_d',20,
                            'frequency to train D')
tf.app.flags.DEFINE_integer('train_D',5,
                            'Number of times to train D')
tf.app.flags.DEFINE_integer('epoches',3000,
                            'number of training epoches')
tf.app.flags.DEFINE_float('lr_',1e-4,
                          'learning rate')
tf.app.flags.DEFINE_float('dlr_',1e-4,
                          'discriminator learning rate')
tf.app.flags.DEFINE_float('lambda_',1.0,
                          'Decaying lambda value')
tf.app.flags.DEFINE_integer('n_filters',256,
                            'number of filters in conv')
tf.app.flags.DEFINE_float('keep_prob',1.0,
                          'keep prob of dropout')

tf.app.flags.DEFINE_integer('checkpoint_step',25,
                            'number of steps before saving checkpoint')

CHECKPOINT_PATH = os.path.join(FLAGS.folder_path,'Checkpoints/')
BASELINE_PATH = os.path.join(FLAGS.folder_path,'baseline/')
SAMPLE_PATH = os.path.join(FLAGS.folder_path,'Samples/')


class Training_config(object):
    #Training configurations
    def __init__(self):
        self.checkpoint_path = FLAGS.check_point
        self.data_path = FLAGS.data_path
        self.batch_size = FLAGS.batch_size
        self.latent_dims = FLAGS.latent_dims
        self.seq_length = FLAGS.seq_length
        self.features_ = FLAGS.features_
        self.features_d = FLAGS.features_d
        self.n_filters = FLAGS.n_filters
        self.epoches = FLAGS.epoches
        self.lr_ = FLAGS.lr_
        self.dlr_ = FLAGS.dlr_
        self.keep_prob = FLAGS.keep_prob
        self.n_resblock = FLAGS.n_resblock
        self.log_dir = FLAGS.log_dir

    def show(self):
        print("Start")

#Generate sample
def encode(data,model):
    new_shape = list(data.shape)
    return model.gen_latent(np.reshape(data,new_shape))

def reconstruct_(model,x,z,x2):
    return model.reconstruct_(x,z,x2)
##########

def z_samples():
    return np.random.normal(0.,1.,size=[FLAGS.batch_size,FLAGS.latent_dims])

def train(training_data,valid_data,data_factory,config):
    vae = WGAN_Model(config)
    num_batch = training_data['A'].shape[0] // FLAGS.batch_size

    #ckpt = tf.train.get_checkpoint_state(CHECKPOINT_PATH + '/model.ckpt-800')
    #if ckpt:
    #vae.load_model(CHECKPOINT_PATH + 'model.ckpt-300')
    #print("Loading model")
    start = 0

    grad_pen_plt = []
    def_grad_plt = []
    p_grad_plt = []

    d_plot = []
    def_dplot = []
    p_dplot = []

    g_plot = []
    def_gplot = []
    p_gplot = []

    v_plot = []
    dv_plot = []
    pv_plot = []

    em_dist_plot = []
    def_em_plot = []
    p_em_plot = []

    total_plot = []

    pen_plot = []
    open_pen_plot = []

    x_axis = []

    for epoch in range(start,FLAGS.epoches):

        d_cost = 0.
        g_cost = 0.
        em_ = 0.
        grad = 0.

        penalty = 0.
        open_penalty = 0.

        def_d = 0.
        def_gen = 0.
        def_em = 0.
        def_grad = 0.
        avg_valid = 0.

        def_valid = 0.

        p_d = 0.
        p_gen = 0.
        p_em = 0.
        p_grad = 0.
        p_valid =0.

        gen_total = 0.

        training_data, valid_data, seq_train, seq_valid, f_train,f_valid,rf_train,rf_valid= data_factory.shuffle()

        batch_id = 0
        if epoch ==0:
            print("pre-train")
            num_d = FLAGS.pretrain_D

        elif epoch%FLAGS.freq_d == 0:
            print("Updating O")
            num_d = 10
        else :
            print("Train O")
            num_d = FLAGS.train_D

        while batch_id < num_batch:
            data_idx = batch_id*FLAGS.batch_size%(training_data['A'].shape[0]-FLAGS.batch_size)
            real_ = training_data['A'][data_idx:data_idx+FLAGS.batch_size]
            real_D = training_data['B'][data_idx:data_idx+FLAGS.batch_size]

            seq_feat = f_train[data_idx:data_idx+FLAGS.batch_size]
            real_ = real_[:,:,[0,1,3,4,5,6,7,8,9,10,11,12]]
            seq_ = seq_train[data_idx:data_idx+FLAGS.batch_size]

            real_feat = rf_train[data_idx:data_idx+FLAGS.batch_size]

            for i in range(num_d):
                d_,em,grad_,\
                d2_,em2,grad2,\
                d3_,em3,grad3 = vae.update_discrim(x=real_,x2=real_D,
                                                        y = seq_,z = z_samples(),
                                                        feat_= seq_feat,feat2_=real_feat
                                                        )

            o_gen,def_gen,play_gen,gen,pen,o_pen = vae.update_gen(real=real_,real_d = real_D,x=seq_,x2 =seq_feat,x3 = real_feat,
                                            z = z_samples())


            # grad += grad_/num_batch
            # d_cost += d_/num_batch
            # g_cost += o_gen/num_batch
            # em_ += em/num_batch


            # def_grad += grad2 / num_batch
            # def_d += d2_/num_batch
            # def_gen += def_gen/num_batch
            # def_em += em2/num_batch

            p_grad += grad3/num_batch
            p_d += d3_/num_batch
            p_gen += play_gen/num_batch
            p_em += em3/num_batch

            gen_total += gen/num_batch
            penalty += pen/num_batch
            open_penalty += o_pen/num_batch

            # validation
            valid_idx = batch_id * FLAGS.batch_size % (valid_data['A'].shape[0] - FLAGS.batch_size)
            valid_ = valid_data['A'][valid_idx:valid_idx + FLAGS.batch_size]
            valid_D = valid_data['B'][valid_idx:valid_idx + FLAGS.batch_size]

            valid_ = valid_[:,:,[0,1,3,4,5,6,7,8,9,10,11,12]]
            seq_v = seq_valid[valid_idx:valid_idx + FLAGS.batch_size]
            rv_feat = f_valid[valid_idx:valid_idx + FLAGS.batch_size]
            rfv_feat = rf_valid[valid_idx:valid_idx + FLAGS.batch_size]

            valid_cost,valid_def,valid_play = vae.valid_loss(x=valid_,x2 = valid_D,
                                        y = seq_v, z = z_samples(),
                                        feat_=rv_feat,feat2_ = rfv_feat)


            avg_valid += valid_cost/num_batch
            def_valid += valid_def/num_batch
            p_valid += valid_play/num_batch

            batch_id += 1

        # plot loss
        x_axis.append(epoch)

        total_plot.append(gen_total)
        grad_pen_plt.append(grad)
        def_grad_plt.append(def_grad)
        p_grad_plt.append(p_grad)

        pen_plot.append(penalty)
        open_pen_plot.append(open_penalty)

        plt.plot(x_axis, pen_plot)
        plt.legend(['Penalty'], loc='upper right')
        plt.savefig('./Data/Penalty.png')
        plt.clf()
        plt.close()

        plt.plot(x_axis, open_pen_plot)
        plt.legend(['open_Penalty'], loc='upper right')
        plt.savefig('./Data/open_Penalty.png')
        plt.clf()
        plt.close()


        em_dist_plot.append(em_)
        plt.plot(x_axis, em_dist_plot)
        plt.legend(['EM_distance'], loc='upper right')
        plt.savefig('./Data/EM_Distance.png')
        plt.clf()
        plt.close()

        def_em_plot.append(def_em)
        plt.plot(x_axis, def_em_plot)
        plt.legend(['EM_distance'], loc='upper right')
        plt.savefig('./Data/Defence_em.png')
        plt.clf()
        plt.close()

        p_em_plot.append(p_em)
        plt.plot(x_axis, p_em_plot)
        plt.legend(['EM_distance'], loc='upper right')
        plt.savefig('./Data/Play_em.png')
        plt.clf()
        plt.close()

        d_plot.append(d_cost)
        plt.plot(x_axis, d_plot, color='b')
        g_plot.append(g_cost)
        plt.plot(x_axis, g_plot, color='g')
        plt.legend(['D_loss', 'G_loss'], loc='lower right')
        plt.savefig('./Data/Discriminator_loss.png')
        plt.clf()
        plt.close()

        def_dplot.append(def_d)
        def_gplot.append(def_gen)
        plt.plot(x_axis,def_dplot,color='b')
        plt.plot(x_axis, def_gplot, color='g')
        plt.legend(['D_loss', 'G_loss'], loc='lower right')
        plt.savefig('./Data/Defence_loss.png')
        plt.clf()
        plt.close()

        p_dplot.append(p_d)
        p_gplot.append(p_gen)
        plt.plot(x_axis, p_dplot, color='b')
        plt.plot(x_axis, p_gplot, color='g')
        plt.legend(['D_loss', 'G_loss'], loc='lower right')
        plt.savefig('./Data/Play_loss.png')
        plt.clf()
        plt.close()

        v_plot.append(avg_valid)
        plt.plot(x_axis, d_plot)
        plt.plot(x_axis, v_plot)
        plt.legend(['D_loss','Valid_loss'], loc='lower right')
        plt.savefig('./Data/Discriminator.png')
        plt.clf()
        plt.close()

        dv_plot.append(def_valid)
        plt.plot(x_axis,def_dplot)
        plt.plot(x_axis,dv_plot)
        plt.legend(['D_loss','Valid_loss'], loc='lower right')
        plt.savefig('./Data/Def_disc.png')
        plt.clf()
        plt.close()

        pv_plot.append(p_valid)
        plt.plot(x_axis, p_dplot)
        plt.plot(x_axis, pv_plot)
        plt.legend(['D_loss', 'Valid_loss'], loc='lower right')
        plt.savefig('./Data/Play_disc.png')
        plt.clf()
        plt.close()

        print("Epoch:", '%04d' % (epoch),
              "\nDiscrim = {}".format(d_cost),
              "\nGen = {}".format(g_cost),
              "\nLambda = {}".format(FLAGS.lambda_))

        if epoch > 0 and epoch % FLAGS.checkpoint_step == 0:
            checkpoint_ = os.path.join(CHECKPOINT_PATH, 'model.ckpt')
            vae.save_model(checkpoint_, epoch)
            print("Saved model")

        if epoch%25 == 0:
            np.save('./Log/d_loss.npy',d_plot)
            np.save('./Log/g_loss.npy',g_plot)
            np.save('./Log/emDist.npy',em_dist_plot)
            np.save('./Log/grad_pen.npy',grad_pen_plt)

            np.save('./Log/defence_D.npy',def_dplot)
            np.save('./Log/defence_G.npy',def_gplot)
            np.save('./Log/defence_em.npy',def_em_plot)
            np.save('./Log/defence_grad.npy',def_grad_plt)


            np.save('./Log/valid_loss.npy',v_plot)
            np.save('./Log/defence_valid.npy',dv_plot)
            np.save('./Log/play_valid.npy',pv_plot)

            np.save('./Log/play_D.npy',p_dplot)
            np.save('./Log/play_G.npy',p_gplot)
            np.save('./Log/play_em.npy',p_em_plot)
            np.save('./Log/play_grad.npy',p_grad_plt)

            #np.save('./Log/penalty.npy',pen_plot)
            #np.save('./Log/open_penalty.npy',open_pen_plot)

            print("Saved Log")

        #Show generated sample
        if epoch % 20 == 0:

            recon= reconstruct_(vae,seq_,z_samples(),seq_feat)
            sample = recon[:,:,:22]
            samples = data_factory.recover_BALL_and_A(sample)
            samples = data_factory.recover_B(samples)
            game_visualizer.plot_data(samples[0], FLAGS.seq_length,
                                      file_path=SAMPLE_PATH + 'reconstruct{}.mp4'.format(epoch),
                                      if_save=True)

    return vae

def main():
    with tf.get_default_graph().as_default() as graph:
        real_data = np.load(FLAGS.data_path+'/F50_D.npy')[:,:FLAGS.seq_length,:,:]
        seq_data = np.load('../Data/50seq2.npy')
        features_ = np.load('../Data/SeqCond.npy')
        real_feat = np.load('../Data/RealCond.npy')

        print("Real Data: ", real_data.shape)
        print("Seq Data: ", seq_data.shape)
        print("Real Feat: ",real_feat.shape)
        print("Seq Feat: ",features_.shape)

        data_factory = DataFactory(real_data =real_data,seq_data = seq_data, features_= features_,real_feat=real_feat)
        training_data, valid_data = data_factory.fetch_data()

        config = Training_config()
        config.show()

        train(training_data,valid_data,data_factory,config)

if __name__ == '__main__':
    main()
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    if not os.path.exists(SAMPLE_PATH):
        os.makedirs(SAMPLE_PATH)
    tf.app.run()