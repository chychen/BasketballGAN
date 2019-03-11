from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from utils import DataFactory
import ops

# https://github.com/hardmaru/diff-vae-tensorflow/blob/master/model.py
# https://medium.com/@anthony_sarkis/tensorboard-quick-start-in-5-minutes-e3ec69f673af
# https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2015/slides/lec13.GAN.pdf

class WGAN_Model():
    def __init__(self, config):
        self.lr_ = config.lr_
        self.dlr_ = config.dlr_
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.latent_dims = config.latent_dims
        self.n_filters = config.n_filters
        self.features_ = config.features_
        self.features_d = config.features_d
        self.keep_prob = config.keep_prob
        self.n_resblock = config.n_resblock
        self.log_dir = config.log_dir

        self.data_factory = DataFactory()

        #Real offence
        self.input_ = tf.placeholder(tf.float32, shape=[None,
                                                        None,
                                                        self.features_],name='Real')
        #Real Defence
        self.input_d = tf.placeholder(tf.float32, shape=[None,
                                                         None,
                                                         self.features_d],name='Real_defence')

        self.ground_feature = tf.placeholder(tf.float32,shape=[None,
                                                               None,
                                                               6],name = 'Real_feat')
        #Condition Data
        self.seq_input = tf.placeholder(tf.float32, shape=[None,
                                                           None,
                                                           self.features_],name = 'Cond_input')

        self.seq_feature = tf.placeholder(tf.float32, shape=[None,
                                                              None,
                                                              6], name='Seq_feat')

        self.z_sample = tf.placeholder(tf.float32, shape=[None, self.latent_dims],name = 'Latent')

        self.network_()
        self.loss_()

        init_ = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init_)
        self.saver = tf.train.Saver(max_to_keep=0)

    def network_(self):
        self.fake_play = self.G_(self.seq_input, self.z_sample, scope='G_', reuse=False)

###########################################################
    def G_(self, cond, x, reuse=False, scope=''):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope(scope, reuse=reuse):
            concat_ = tf.concat([cond, self.seq_feature], axis=-1)

            with tf.variable_scope('conds_linear') as scope:
                conds_linear = layers.fully_connected(
                    inputs=concat_,
                    num_outputs=self.n_filters,
                    activation_fn=None,
                    weights_initializer=layers.xavier_initializer(uniform=False),
                    biases_initializer=tf.zeros_initializer(),
                    scope=scope
                )

            with tf.variable_scope('latents_linear') as scope:
                # linear projection latents to hyper space
                latents_linear = layers.fully_connected(
                    inputs=x,
                    num_outputs=self.n_filters,
                    activation_fn=None,
                    weights_initializer=layers.xavier_initializer(
                        uniform=False),
                    biases_initializer=tf.zeros_initializer(),
                    scope=scope
                )
            latents_linear = tf.reshape(latents_linear, shape=[
                -1, 1, self.n_filters])

            next_input = conds_linear + latents_linear

            for i in range(self.n_resblock):
                res_b = ops.res_block('G_Res' + str(i),
                                      next_input,
                                      n_filters=self.n_filters,
                                      n_layers=2,
                                      residual_alpha=1.0,
                                      )
                next_input = res_b

            with tf.variable_scope('conv_result') as scope:
                normed = layers.layer_norm(next_input)
                nonlinear = ops.leaky_relu(normed)
                conv_result = tf.layers.conv1d(
                    inputs=nonlinear,
                    filters= 28,
                    kernel_size=5,
                    strides=1,
                    padding='same',
                    activation=None,
                    kernel_initializer=layers.xavier_initializer(),
                    bias_initializer=tf.zeros_initializer()
                )

            return conv_result

###########################################################
    def discriminator(self, conds, x, reuse=False, scope=''):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        concat_ = tf.concat([conds, x], axis=-1)

        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope('conv_input') as scope:
                conv_input = tf.layers.conv1d(
                    inputs=concat_,
                    filters=self.n_filters,
                    kernel_size=5,
                    strides=1,
                    padding='same',
                    activation=ops.leaky_relu,
                    kernel_initializer=layers.xavier_initializer(),
                    bias_initializer=tf.zeros_initializer()
                )

            next_input = conv_input

            for i in range(self.n_resblock):
                res_b = ops.res_block('disc_Res' + str(i),
                                      next_input,
                                      n_filters=self.n_filters,
                                      n_layers=2,
                                      residual_alpha=1.0,
                                      )
                next_input = res_b

            with tf.variable_scope('conv_output') as scope:
                normed = layers.layer_norm(next_input)
                nonlinear = ops.leaky_relu(normed)
                conv_output = tf.layers.conv1d(
                    inputs=nonlinear,
                    filters=1,
                    kernel_size=5,
                    strides=1,
                    padding='same',
                    activation=ops.leaky_relu,
                    kernel_initializer=layers.xavier_initializer(),
                    bias_initializer=tf.zeros_initializer()
                )
                score = conv_output
                conv_output = tf.reduce_mean(conv_output, axis=1)

                final_ = tf.reshape(conv_output, shape=[-1])

            return final_, score

    def loss_(self):
        real_play = tf.concat([self.input_, self.input_d], axis=-1)
        real_play = tf.concat([real_play,self.ground_feature],axis=-1)

        fake_play = self.fake_play
        offence = tf.reshape(self.fake_play[:, :,:12], shape =[self.batch_size,self.seq_length,12])
        G_feat = tf.reshape(self.fake_play[:, :,22:], shape =[self.batch_size,self.seq_length,6])
        G_offence = tf.concat([offence,G_feat],axis=-1)

        defence = tf.reshape(self.fake_play[:, :,12:22], shape =[self.batch_size,self.seq_length,10])

        #conds = self.seq_input
        conds = tf.concat([self.seq_input, self.seq_feature], axis=-1)
        #offence Discriminator - Condition defence
        with tf.variable_scope(tf.get_variable_scope()):
            o_input = tf.concat([self.input_,self.ground_feature],axis=-1)
            real_o, self.real_score = self.discriminator(defence, o_input, reuse=False, scope='O_disc')
            fake_o, self.fake_score = self.discriminator(defence, G_offence, reuse=True, scope='O_disc')

            self.d_cost, self.grad_pen = self.loss_d(defence,o_input, real_o,
                                                     G_offence, fake_o,scope='O_disc')

            self.em_dist = tf.reduce_mean(real_o) - tf.reduce_mean(fake_o)
        #Defence Discriminator - Condition offence
        with tf.variable_scope(tf.get_variable_scope()):
            real_d, self.real_scoreD = self.discriminator(o_input, self.input_d, reuse=False, scope='D_disc')
            fake_d, self.fake_scoreD = self.discriminator(o_input, defence, reuse=True, scope='D_disc')

            self.d2_cost, self.grad2_pen = self.loss_d(o_input,self.input_d, real_d,
                                                       defence, fake_d,scope='D_disc')

            self.em2_dist = tf.reduce_mean(real_d) - tf.reduce_mean(fake_d)

        #Full play Discriminator
        with tf.variable_scope(tf.get_variable_scope()):
            real_p, self.real_scorep = self.discriminator(conds, real_play, reuse=False, scope='P_disc')
            fake_p, self.fake_scorep = self.discriminator(conds, fake_play, reuse=True, scope='P_disc')

            self.d3_cost, self.grad3_pen = self.loss_d(conds,real_play, real_p,
                                                       fake_play, fake_p,scope='P_disc')

            self.em3_dist = tf.reduce_mean(real_p) - tf.reduce_mean(fake_p)

        # wgan gen loss
        self.g_cost = -tf.reduce_mean(fake_o)
        self.g2_cost = -tf.reduce_mean(fake_d)
        self.g3_cost = -tf.reduce_mean(fake_p)

        self.realDisc = tf.reduce_mean(real_o)
        self.realDisc2 = tf.reduce_mean(real_d)
        self.realDisc3 = tf.reduce_mean(real_p)

        scale = tf.abs(self.g_cost)
        scale2 = tf.abs(self.g2_cost)
        #self.penalty = self.dribbler_penalty(self.fake_play,real_play)
        #self.open_penalty = self._open_shot_penalty(real_play,self.fake_play)

        self.gen_cost = (0.9*self.g_cost)+(0.9*(self.g2_cost))+self.g3_cost #+(scale*self.penalty)+(scale2*self.open_penalty)


        self.t_vars = tf.trainable_variables()
        self.gen_vars = [var for var in self.t_vars if 'G_' in var.name]

        self.dis_vars = [var for var in self.t_vars if 'O_disc' in var.name]
        self.dis2_vars = [var for var in self.t_vars if 'D_disc' in var.name]
        self.dis3_vars = [var for var in self.t_vars if 'P_disc' in var.name]

        # ADAM optimizer
        self.o_optimizer = tf.train.AdamOptimizer(self.lr_, beta1=0.5, beta2=0.9).minimize(self.d_cost,
                                                                                           var_list=self.dis_vars)
        self.d_optimizer = tf.train.AdamOptimizer(self.lr_, beta1=0.5, beta2=0.9).minimize(self.d2_cost,
                                                                                           var_list=self.dis2_vars)
        self.p_optimizer = tf.train.AdamOptimizer(self.lr_, beta1=0.5, beta2=0.9).minimize(self.d3_cost,
                                                                                           var_list=self.dis3_vars)

        self.genO_optimizer = tf.train.AdamOptimizer(self.lr_, beta1=0.5, beta2=0.9).minimize(self.gen_cost,
                                                                                             var_list=self.gen_vars)


    def loss_d(self, conds, real_sample, real, G_sample, fake,scope):

        epsilon = tf.random_uniform(
            [self.batch_size, 1, 1], minval=0.0, maxval=1.0)

        X_inter = epsilon * real_sample + (1.0 - epsilon) * G_sample

        g_d, _ = self.discriminator(conds, X_inter, reuse=True, scope= scope)

        grad = tf.gradients(g_d, [X_inter])[0]

        sum_ = tf.reduce_sum(tf.square(grad), axis=[1, 2])

        grad_norm = tf.sqrt(sum_)
        grad_pen = 10.0 * tf.reduce_mean(
            tf.square(grad_norm - 1.0))

        f_fake = tf.reduce_mean(fake)
        f_real = tf.reduce_mean(real)

        loss = f_fake - f_real + grad_pen

        return loss, grad_pen

    def dribbler_penalty(self, fake,real):
        fake_ = self._dribbler_score(fake, log_scope_name='fake_pen')

        real_ = self._dribbler_score(real,log_scope_name='real_pen')

        return tf.abs(real_ - fake_)

    def _dribbler_score(self, inputs, log_scope_name=''):
        # Get clostest distance between player and ball
        # min Real distance minus min fake distance
        with tf.name_scope('dribbler_score') as scope:
            # ball x and y pos
            basket_right_x = tf.constant(self.data_factory.BASKET_RIGHT[0], dtype=tf.float32, shape=
                                            [self.batch_size, self.seq_length, 1, 1])
            basket_right_y = tf.constant(self.data_factory.BASKET_RIGHT[1], dtype=tf.float32, shape=
                                            [self.batch_size, self.seq_length, 1, 1])
            basket_pos = tf.concat([basket_right_x, basket_right_y], axis=-1)

            ball_pos = tf.reshape(inputs[:, :, :2], shape=[
                self.batch_size, self.seq_length, 1, 2])

            feat_ = tf.reshape(inputs[:,:,22:],
                               shape=[self.batch_size,self.seq_length,6])
            # players x and y pos
            teamB_pos = tf.reshape(
                inputs[:, :, 2:12], shape=[self.batch_size, self.seq_length, 5, 2])

            teamB_pos = tf.concat([teamB_pos,basket_pos],axis = 2)

            vec_ball = ball_pos - teamB_pos
            dist_ = tf.norm(vec_ball, ord='euclidean', axis=-1)
            dist_f = tf.multiply(dist_,tf.round(feat_))
            dribbler_scMin = tf.reduce_max(dist_f, axis=-1)
            dribbler_sc = tf.reduce_mean(dribbler_scMin)

            return dribbler_sc

    def _open_shot_penalty(self,real,fake):
        real_o = tf.reshape(real[:,:,:12],shape = [
            self.batch_size,self.seq_length,12])
        real_d = tf.reshape(real[:,:,12:22],shape =[
            self.batch_size,self.seq_length,10])

        fake_o = tf.reshape(fake[:,:,:12],shape=[
            self.batch_size,self.seq_length,12])
        fake_d = tf.reshape(fake[:, :, 12:22], shape=[
            self.batch_size, self.seq_length, 10])

        real_penalty = self._open_shot_score(real_o,real_d)

        fake_penalty = self._open_shot_score(fake_o,fake_d)


        return tf.abs(real_penalty-fake_penalty)

    def _open_shot_score(self,offence_,defence_):
        with tf.name_scope('wide_open_score') as scope:
            # ball x and y pos
            ball_pos = tf.reshape(offence_[:, :, :2], shape=[
                self.batch_size, self.seq_length, 1, 2])
            # players x and y pos
            teamB_pos = tf.reshape(
                defence_, shape=[self.batch_size, self.seq_length, 5, 2])
            basket_right_x = tf.constant(self.data_factory.BASKET_RIGHT[0],dtype = tf.float32,shape=
                                         [self.batch_size,self.seq_length,1,1])
            basket_right_y = tf.constant(self.data_factory.BASKET_RIGHT[1], dtype=tf.float32, shape=
                                        [self.batch_size, self.seq_length, 1, 1])
            basket_pos = tf.concat([basket_right_x,basket_right_y],axis=-1)
            vec_ball_2_team = ball_pos - teamB_pos
            vec_ball_2_basket = ball_pos - basket_pos
            b2teamB_dot_b2basket = tf.matmul(vec_ball_2_team,vec_ball_2_basket,transpose_b=True)
            b2teamB_dot_b2basket = tf.reshape(b2teamB_dot_b2basket,shape=[
                                    self.batch_size,self.seq_length,5])

            dist_teamB = tf.norm(vec_ball_2_team, ord='euclidean', axis=-1)
            dist_basket = tf.norm(vec_ball_2_basket, ord='euclidean', axis=-1)

            theta = tf.acos(b2teamB_dot_b2basket/
                            (dist_teamB*dist_basket+1e-3))
            open_shot_score_all = (theta+1.0)*(dist_teamB+1.0)
            open_shot_score_min = tf.reduce_min(open_shot_score_all,axis=-1)
            open_shot_score = tf.reduce_mean(open_shot_score_min)

            too_close_penalty = 0.0
            for i in range(5):
                vec = tf.subtract((teamB_pos[:,:,i:i+1]),teamB_pos)
                dist = tf.sqrt((vec[:,:,:,0]+1e-8)**2+(vec[:,:,:,1]+1e-8)**2)
                too_close_penalty -= tf.reduce_mean(dist)

            return open_shot_score+too_close_penalty


    def update_discrim(self, x,x2, y, feat_,feat2_, z):
        train_feed = {self.input_: x,self.input_d:x2,
                      self.seq_input: y, self.seq_feature: feat_,
                      self.z_sample: z,self.ground_feature:feat2_
                      }

        d_cost, real_score, grad, em,d_opt = self.sess.run((self.d_cost,
                                                              self.realDisc, self.grad_pen, self.em_dist,
                                                                self.o_optimizer),
                                                              feed_dict=train_feed)

        d2_cost, real2_score, grad2, em2, d2_opt = self.sess.run((self.d2_cost, self.realDisc2, self.grad2_pen,
                                                                  self.em2_dist,self.d_optimizer),
                                                                  feed_dict=train_feed)

        d3_cost, real3_score, grad3, em3, d3_opt = self.sess.run((self.d3_cost, self.realDisc3, self.grad3_pen,
                                                                  self.em3_dist,self.p_optimizer),
                                                                  feed_dict=train_feed)

        return d_cost,grad,em,\
               d2_cost, grad2, em2,\
               d3_cost,grad3,em3

    def update_gen(self,real,real_d, x, x2,x3, z):
        train_feed = {self.input_:real,self.input_d:real_d,self.seq_input: x, self.seq_feature: x2,self.ground_feature:x3, self.z_sample: z}

        g_opt,g_cost,def_cost,p_cost,gen_cost,pen,open_pen = self.sess.run((self.genO_optimizer,self.g_cost,self.g2_cost,self.g3_cost,self.gen_cost,self.penalty,self.open_penalty                                                    ),
                                                    feed_dict=train_feed)

        return g_cost,def_cost,p_cost, gen_cost,pen,open_pen

    def valid_loss(self, x, x2, y, feat_,feat2_, z):
        train_feed = {self.input_: x, self.input_d: x2,
                      self.seq_input: y, self.seq_feature: feat_,
                      self.z_sample: z,self.ground_feature: feat2_}

        valid_cost = self.sess.run((self.d_cost), feed_dict=train_feed)

        valid2_cost = self.sess.run((self.d2_cost), feed_dict=train_feed)

        valid3_cost = self.sess.run((self.d3_cost), feed_dict=train_feed)

        return valid_cost,valid2_cost,valid3_cost

    # Sampling
    def reconstruct_(self, x, z, x2):
        return self.sess.run(self.fake_play,
                             feed_dict={self.seq_input: x,
                                        self.z_sample: z, self.seq_feature: x2})

    def get_score(self,sample,feat_,seq):

        train_feed = {self.fake_play:sample,
                      self.seq_feature:feat_,
                      self.seq_input:seq}

        return self.sess.run(self.fake_scorep,feed_dict=train_feed)



    def save_model(self, checkpoint_path, epoch):
        self.saver.save(self.sess, checkpoint_path, global_step=epoch)

    def load_model(self, checkpoint_path):
        self.saver.restore(self.sess, checkpoint_path)
