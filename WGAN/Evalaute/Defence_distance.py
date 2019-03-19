import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd

BASKET_RIGHT = [90, 25]


def distance(x1,y1,x2,y2):
    dist = math.sqrt( ((x2-x1)**2)+((y2-y1)**2))
    return dist

def nearest_defender(player,teamB):
    dist_ = []

    for x in range(5):
        defender_ = teamB[x]
        dist = distance(player[0],player[1],defender_[0],defender_[1])
        dist_.append(dist)
    dist_min = np.min(dist_)

    return dist_min
DATA_PATH = './Data/Samples'

real = np.load(DATA_PATH+'/Real/RealTrain.npy')
batch = np.load(DATA_PATH+'/WithPen/Train_Samples.npy')

no_pen = np.load(DATA_PATH+'/NoPen/Train_Samples.npy')
vae = np.load(DATA_PATH+'/VAE/Train_VAESamples.npy')
wgan = np.load('./Data/Samples/WGAN/Train_Samples.npy')
basketb_right = [90, 25]

#feat_ = np.round(sample2[:,:,22:])
#feat_ = feat_.astype(int)

feat_ = np.load('./Data/Samples/Real/RealCondition.npy')

teamA_ = real[:,:,2:12]
teamB_ = real[:,:,12:22]

teamA_ = np.reshape(teamA_,newshape=[len(real),50,5,2])
teamB_ = np.reshape(teamB_,newshape=[len(real),50,5,2])

genA = np.reshape(batch[:,:,2:12],newshape=[len(real),50,5,2])
genB = np.reshape(batch[:,:,12:22],newshape=[len(real),50,5,2])
genW_ = np.round(batch[:,:,22:])
featG_ = genW_.astype(int)

penA = np.reshape(no_pen[:,:,2:12],newshape=[len(real),50,5,2])
penB = np.reshape(no_pen[:,:,12:22],newshape=[len(real),50,5,2])
penW_ = np.round(no_pen[:,:,22:])
featP_ = penW_.astype(int)

vaeA = np.reshape(vae[:,:,2:12],newshape=[len(real),50,5,2])
vaeB = np.reshape(vae[:,:,12:22],newshape=[len(real),50,5,2])
featV_ = np.round(vae[:,:,22:])
featV_ = featV_.astype(int)

wganA = np.reshape(wgan[:,:,2:12],newshape=[len(real),50,5,2])
wganB = np.reshape(wgan[:,:,12:22],newshape=[len(real),50,5,2])
featW_ = np.round(wgan[:,:,22:])
featW_ = featW_.astype(int)

pass_def = [0,0,0,0,0]
pass_ = 0

total_distance = 0

dist_sample = []
distGen_sample = []
distVae_sample = []
distWgan_sample = []
distPen_sample = []

for n in range(len(real)):
    test = feat_[n]

    testG = featG_[n]
    testP = featP_[n]
    testV = featV_[n]
    testW = featW_[n]

    defender_dist = 0

    teamB_defender = teamB_[n]
    player_ = teamA_[n]

    teamvae_defender = vaeB[n]
    playervae = vaeA[n]

    teamWgan_defender = wganB[n]
    playerWgan = wganA[n]

    teamGen_defender = genB[n]
    playerGen = genA[n]

    teamPen_defender = penB[n]
    playerPen = penA[n]

    possession_ = 1

    for x in range(50):
        playerA_ = player_[x]
        playerB_ = teamB_defender[x]

        if not (test[x] == pass_def).all():
            for i in range(5):
                if test[x,i] == 1:
                    dist = nearest_defender(playerA_[i],playerB_)

                    dist_sample.append(np.floor(dist))

        playerA_ = playervae[x]
        playerB_ = teamvae_defender[x]

        if not (testV[x] == pass_def).all():
            for i in range(5):
                if testV[x, i] == 1:
                    distVae = nearest_defender(playerA_[i], playerB_)
                    distVae_sample.append(np.floor(distVae))

                    #defender_dist += dist
                    #possession_ += 1
        playerA_ = playerWgan[x]
        playerB_ = teamWgan_defender[x]

        if not (testW[x] == pass_def).all():
            for i in range(5):
                if testW[x, i] == 1:
                    dist = nearest_defender(playerA_[i], playerB_)
                    distWgan_sample.append(np.floor(dist))

        playerA_ = playerGen[x]
        playerB_ = teamGen_defender[x]

        if not (testG[x] == pass_def).all():
            for i in range(5):
                if testG[x, i] == 1:
                    dist = nearest_defender(playerA_[i], playerB_)
                    distGen_sample.append(np.floor(dist))

        playerA_ = playerPen[x]
        playerB_ = teamPen_defender[x]

        if not (testP[x] == pass_def).all():
            for i in range(5):
                if testP[x, i] == 1:
                    dist = nearest_defender(playerA_[i], playerB_)
                    distPen_sample.append(np.floor(dist))

        #total_distance += defender_dist/possession_

print("Real: ",np.mean(dist_sample))
print("BGAN: ",np.mean(distGen_sample))
print("No Pen: ",np.mean(distPen_sample))
print("VAE: ",np.mean(distVae_sample))
print("WGAN: ",np.mean(distWgan_sample))
print(" ")
print("Real: ",np.std(dist_sample))
print("BGAN: ",np.std(distGen_sample))
print("No Pen: ",np.std(distPen_sample))
print("VAE: ",np.std(distVae_sample))
print("WGAN: ",np.std(distWgan_sample))

'''
x_axis = np.linspace(1,20,20)
y_axis = []
y2_axis = []
y3_axis = []
y4_axis = []
y5_axis = []

count = 0
for i in range(0,20):
    for x in dist_sample:
        if x == i:
            count += 1

    percent_ = (count/len(dist_sample))*100
    y_axis.append(percent_)
    count = 0

    for x in distVae_sample:
        if x == i:
            count += 1

    percent_ = (count/len(distVae_sample))*100
    y4_axis.append(percent_)
    count = 0

    for x in distWgan_sample:
        if x == i:
            count += 1

    percent_ = (count/len(distWgan_sample))*100
    y3_axis.append(percent_)
    count = 0

    for x in distGen_sample:
        if x == i:
            count += 1

    percent_ = (count/len(distGen_sample))*100
    y2_axis.append(percent_)
    count = 0

    for x in distPen_sample:
        if x == i:
            count += 1

    percent_ = (count/len(distPen_sample))*100
    y5_axis.append(percent_)
    count = 0


plt.plot(x_axis,y_axis)
plt.plot(x_axis,y2_axis)
plt.plot(x_axis,y5_axis)
plt.plot(x_axis,y4_axis)
plt.plot(x_axis,y3_axis)


plt.legend(['Real','Gen','NoPen','VAE','WGAN'])
plt.show()
'''




'''
offence_ = tf.placeholder(tf.float32, shape=[None,
                           None,
                           12])

defence_ = tf.placeholder(tf.float32, shape=[None,
                           None,
                           10])

feed_dict = { offence_ : teamA_,
              defence_ : teamB_,
            }

with tf.get_default_graph().as_default() as graph:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with tf.name_scope('wide_open_score') as scope:
            # ball x and y pos
            ball_pos = tf.reshape(offence_[:, :, :2], shape=[
                len(real), 50, 1, 2])
            # players x and y pos
            teamB_pos = tf.reshape(
                defence_, shape=[len(real), 50, 5, 2])

            basket_right_x = tf.constant(BASKET_RIGHT[0], dtype=tf.float32, shape=
            [len(real), 50, 1, 1])
            basket_right_y = tf.constant(BASKET_RIGHT[1], dtype=tf.float32, shape=
            [len(real), 50, 1, 1])

            basket_pos = tf.concat([basket_right_x, basket_right_y], axis=-1)
            vec_ball_2_team = ball_pos - teamB_pos
            vec_ball_2_basket = ball_pos - basket_pos
            b2teamB_dot_b2basket = tf.matmul(vec_ball_2_team, vec_ball_2_basket, transpose_b=True)
            b2teamB_dot_b2basket = tf.reshape(b2teamB_dot_b2basket, shape=[
                len(real),50, 5])

            dist_teamB = tf.norm(vec_ball_2_team, ord='euclidean', axis=-1)
            dist_basket = tf.norm(vec_ball_2_basket, ord='euclidean', axis=-1)

            theta = tf.acos(b2teamB_dot_b2basket /
                            (dist_teamB * dist_basket + 1e-3))
            open_shot_score_all = (theta + 1.0) * (dist_teamB + 1.0)
            open_shot_score_min = tf.reduce_min(open_shot_score_all, axis=-1)
            open_shot_score = tf.reduce_mean(open_shot_score_min)


        result = sess.run(
            open_shot_score_min, feed_dict=feed_dict)

        print(result.shape)

        result = np.rint(result)

        print(result)

        count = 0
        x_axis = np.linspace(1,10,9)
        y_axis = []
        for i in range(1,10):
            for j in range(len(result)):
                for x in result[j]:
                    if x == i:
                        count += 1
            percent_ = (count/(len(result)*50))*100
            y_axis.append(percent_)
            count = 0

        plt.plot(x_axis,y_axis)
        plt.show()
'''

