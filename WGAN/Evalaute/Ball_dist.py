import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

DATA_PATH = '../Data/'

#sample = np.load('./Data/Diversity/results_100.npy')[0]
#length_ = np.load('./Data/Diversity/result_length.npy')


#sample = np.load('./Data/Samples/NoPen/Train_Samples.npy')
#sample2 = np.load('./Data/Samples/WithPen/Train_Samples.npy')

real_feat = np.load('../Data/Valid/ValidCondition.npy')

#Real
real = np.load(DATA_PATH + '/Valid/RealValid.npy')
#Basketball GAN
#batch = np.load(DATA_PATH+'/Valid/Valid/Diff/Valid_ReconDiff.npy')
sample2 = np.load(DATA_PATH+'/Final/Valid_ReconDiff_v2.npy')
#BasketballGAN2_WGAN
wgan = np.load('../Data/Valid/Valid/Valid_SamplesWgan.npy')
#VAE
vae = np.load(DATA_PATH + '/Valid/Valid_VAESamples.npy')
#condition
seq = np.load('../Data/Valid/Valid_SamplesWgan.npy')
#seq = np.load('../Data/Valid/Valid_ConditionData.npy')



#with Penalty
featB_ = np.round(sample2[:,:,22:27])
featB_ = featB_.astype(int)
teamB_ = sample2[:,:,2:12]
BallB_ = sample2[:,:,:2]
teamB_ = np.reshape(teamB_,newshape=[len(sample2),50,5,2])

#Real
teamReal_ = real[:,:,2:12]
Ballreal_ = real[:,:,:2]
teamReal_ = np.reshape(teamReal_,newshape=[len(real),50,5,2])

#VAE
featV_ = np.round(vae[:,:,22:27])
featV_ = featV_.astype(int)
teamV_ = vae[:,:,2:12]
BallV_ = vae[:,:,:2]
teamV_ = np.reshape(teamV_,newshape=[len(vae),50,5,2])

#WGAN
featW_ = np.round(wgan[:,:,22:27])
featW_ = featW_.astype(int)
teamW_ = wgan[:,:,2:12]
BallW_ = wgan[:,:,:2]
teamW_ = np.reshape(teamW_,newshape=[len(wgan),50,5,2])
##################################
pass_def = [0,0,0,0,0]
pass_ = 0

total_distance = 0

dist_sample = []
distB_sample = []
distReal_sample = []
distVae_sample = []
distWgan_sample = []

for n in range(len(real)):
    testB = featB_[n]
    testR = real_feat[n]
    testV = featV_[n]
    testW = featW_[n]

    #no penalty
    #ball = Ball_[n]
    #player_ = teamA_[n]

    #with penalty
    ballB = BallB_[n]
    playerB_ = teamB_[n]

    #real
    ballR = Ballreal_[n]
    playerR_ = teamReal_[n]

    #vae
    ballV = BallV_[n]
    playerV_ = teamV_[n]

    #wgan
    ballW = BallW_[n]
    playerW_ = teamW_[n]
    #l = length_[n]
    for x in range(50):
        #no penalty
        '''
        playerA_ = player_[x]
        ball_ = ball[x]
        if not (test[x] == pass_def).all():
            for i in range(5):
                if test[x,i] == 1:
                    playerA2_ = playerA_[i]
                    dist = distance(playerA2_[0], playerA2_[1], ball_[0], ball_[1])
                    dist_sample.append(np.floor(dist))
        '''

        #with penalty
        playerb_ = playerB_[x]
        ballB_ = ballB[x]
        if not (testB[x] == pass_def).all():
            for i in range(5):
                if testB[x,i] == 1:
                    playerA2_ = playerb_[i]
                    dist = distance(playerA2_[0], playerA2_[1], ballB_[0], ballB_[1])
                    distB_sample.append(np.floor(dist))

        #real
        playerR = playerR_[x]
        ballr_ = ballR[x]
        if not (testR[x] == pass_def).all():
            for i in range(5):
                if testR[x, i] == 1:
                    playerA2_ = playerR[i]
                    dist = distance(playerA2_[0], playerA2_[1], ballr_[0], ballr_[1])
                    distReal_sample.append(np.floor(dist))


        #vae
        playerV = playerV_[x]
        ballv_ = ballV[x]
        if not (testV[x] == pass_def).all():
            for i in range(5):
                if testV[x, i] == 1:
                    playerA2_ = playerV[i]
                    dist = distance(playerA2_[0], playerA2_[1], ballv_[0], ballv_[1])
                    distVae_sample.append(np.floor(dist))

        #Wgan
        playerW = playerW_[x]
        ballw_ = ballW[x]
        if not (testW[x] == pass_def).all():
            for i in range(5):
                if testW[x, i] == 1:
                    playerA2_ = playerW[i]
                    dist = distance(playerA2_[0], playerA2_[1], ballw_[0], ballw_[1])
                    distWgan_sample.append(np.floor(dist))

count = 0
x_axis = np.linspace(0,10,11)


y_axis = []
y2_axis = []
y3_axis = []
y4_axis = []
y5_axis = []

for i in range(11):
    ######
    '''
    for x in dist_sample:
        if x == i:
            count += 1
    percent_ = (count/len(dist_sample))*100
    y_axis.append(percent_)
    count = 0
    '''

    ########
    for x in distB_sample:
        if x == i:
            count += 1

    percent_ = (count/len(distB_sample))*100
    y2_axis.append(percent_)
    count = 0

    ########
    for x in distReal_sample:
        if x == i:
            count += 1

    percent_ = (count/len(distReal_sample))*100
    y_axis.append(percent_)
    count = 0

    ########
    for x in distVae_sample:
        if x == i:
            count += 1
    percent_ = (count/len(distVae_sample))*100
    y4_axis.append(percent_)
    count = 0

    ######
    for x in distWgan_sample:
        if x == i:
            count += 1

    percent_ = (count/len(distWgan_sample))*100
    y5_axis.append(percent_)
    count = 0

sns.set(rc={'lines.linewidth':1.7})
sns.set_style("darkgrid")
#sns.set_palette("Set1")
#sns.set_palette(['C6','C3','C1','C5','C9'])


#plt.plot(x_axis,y3_axis,c = 'b')

plt.plot(x_axis,y_axis,c='b')

plt.plot(x_axis,y2_axis,c='g')

plt.plot(x_axis,y5_axis,c='purple')
plt.plot(x_axis,y4_axis,c='orange')

plt.xlabel('ft')
plt.ylabel('Probability %')
plt.title('Ball Handler to Ball Distance')

plt.legend(['Real','BasketballGAN','BasketballGAN2','C-VAE'])

plt.savefig('../../../Graphs/BallDist.svg')
#plt.show()
