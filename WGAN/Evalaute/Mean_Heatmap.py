import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1 import AxesGrid


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
DATA_PATH = '../Data/'

real = np.load(DATA_PATH + '/Valid/RealValid.npy')
#Basketball GAN
#batch = np.load(DATA_PATH+'/Valid/Valid/Diff/Valid_ReconDiff.npy')
batch = np.load(DATA_PATH+'/Final/Valid_ReconDiff_v2.npy')
#BasketballGAN2_WGAN
wgan = np.load('../Data/Valid/Valid/Valid_SamplesWgan.npy')
#VAE
vae = np.load(DATA_PATH + '/Valid/Valid_VAESamples.npy')
#condition
seq = np.load('../Data/Valid/Valid_SamplesWgan.npy')

basketb_right = [90, 25]

feat_ = np.load('../Data/Valid/ValidCondition.npy')

teamA_ = real[:,:,2:12]
teamB_ = real[:,:,12:22]

teamA_ = np.reshape(teamA_,newshape=[len(real),50,5,2])
teamB_ = np.reshape(teamB_,newshape=[len(real),50,5,2])

genA = np.reshape(batch[:,:,2:12],newshape=[len(real),50,5,2])
genB = np.reshape(batch[:,:,12:22],newshape=[len(real),50,5,2])
genW_ = np.round(batch[:,:,22:27])
featG_ = genW_.astype(int)

vaeA = np.reshape(vae[:,:,2:12],newshape=[len(real),50,5,2])
vaeB = np.reshape(vae[:,:,12:22],newshape=[len(real),50,5,2])
featV_ = np.round(vae[:,:,22:27])
featV_ = featV_.astype(int)

wganA = np.reshape(wgan[:,:,2:12],newshape=[len(real),50,5,2])
wganB = np.reshape(wgan[:,:,12:22],newshape=[len(real),50,5,2])
featW_ = np.round(wgan[:,:,22:27])
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
    testV = featV_[n]
    testW = featW_[n]

    teamB_defender = teamB_[n]
    player_ = teamA_[n]

    teamvae_defender = vaeB[n]
    playervae = vaeA[n]

    teamWgan_defender = wganB[n]
    playerWgan = wganA[n]

    teamGen_defender = genB[n]
    playerGen = genA[n]


    for x in range(50):
        playerA_ = player_[x]
        playerB_ = teamB_defender[x]

        if not (test[x] == pass_def).all():
            for i in range(5):
                if test[x, i] == 1:
                    dist = nearest_defender(playerA_[i], playerB_)
                    dist_sample.append([playerA_[i][0], playerA_[i][1], np.floor(dist)])


        playerA_ = playervae[x]
        playerB_ = teamvae_defender[x]
    
        if not (testV[x] == pass_def).all():
            for i in range(5):
                if testV[x, i] == 1:
                    dist = nearest_defender(playerA_[i], playerB_)
                    distVae_sample.append([playerA_[i][0], playerA_[i][1], np.floor(dist)])
    
                    # defender_dist += dist
                    # possession_ += 1
        playerA_ = playerWgan[x]
        playerB_ = teamWgan_defender[x]
    
        if not (testW[x] == pass_def).all():
            for i in range(5):
                if testW[x, i] == 1:
                    dist = nearest_defender(playerA_[i], playerB_)
                    distWgan_sample.append([playerA_[i][0],playerA_[i][1],np.floor(dist)])

        playerA_ = playerGen[x]
        playerB_ = teamGen_defender[x]
    
        if not (testG[x] == pass_def).all():
            for i in range(5):
                if testG[x, i] == 1:
                    dist = nearest_defender(playerA_[i], playerB_)
                    distGen_sample.append([playerA_[i][0], playerA_[i][1], np.floor(dist)])


r_ = np.array(dist_sample)
wgan_ = np.array(distWgan_sample)
gen_ = np.array(distGen_sample)
vae_ = np.array(distVae_sample)

fig = plt.figure(figsize=(10,6))

axes = AxesGrid(fig,111,
                nrows_ncols=(2,2),
                axes_pad=0.5,
                cbar_mode='single',
                cbar_location='right')


im = axes[0].imshow(np.random.random((16, 16)), cmap='viridis',
                   vmin=0, vmax=1)

ax1 = axes[0]
ax1.set_xlim(47, 94)
ax1.set_ylim(50, 0)
ax1.axis("off")
ax1.set_title('Real')

ax2 = axes[2]
ax2.set_xlim(47, 94)
ax2.set_ylim(50,0)
ax2.axis("off")
ax2.set_title('BasketballGAN2')

ax3 = axes[1]
ax3.set_xlim(47, 94)
ax3.set_ylim(50,0)
ax3.axis("off")
ax3.set_title('BasketballGAN')

ax4 = axes[3]
ax4.set_xlim(47, 94)
ax4.set_ylim(50,0)
ax4.axis("off")
ax4.set_title('VAE')

df = pd.DataFrame({'x_axis': r_[:,0],
                   'y_axis': r_[:,1],
                   'dist': r_[:,2]
                   })

df2 = pd.DataFrame({'x_axis': wgan_[:,0],
                   'y_axis': wgan_[:,1],
                   'dist': wgan_[:,2]
                   })
df3 = pd.DataFrame({'x_axis': gen_[:,0],
                   'y_axis': gen_[:,1],
                   'dist': gen_[:,2]
                   })
df4 = pd.DataFrame({'x_axis': vae_[:,0],
                   'y_axis': vae_[:,1],
                   'dist': vae_[:,2]
                   })

max = 10


im = plt.imshow(np.random.random((0,max)), vmin=0, vmax=max,aspect="auto")


ax1.hexbin(x=df['x_axis'],y = df['y_axis']
           ,C = df['dist'],edgecolors=None,
           reduce_C_function= np.mean,
           gridsize=25,alpha=0.7,vmax=max,vmin=0)


ax2.hexbin(x=df2['x_axis'],y = df2['y_axis']
           ,C = df2['dist'],edgecolors=None,
           reduce_C_function= np.mean,
           gridsize=25,alpha=0.7,vmax=max,vmin=0)

ax3.hexbin(x=df3['x_axis'],y = df3['y_axis']
           ,C = df3['dist'],edgecolors=None,
           reduce_C_function= np.mean,
           gridsize=25,alpha=0.7,vmax=max,vmin=0)
ax4.hexbin(x=df4['x_axis'],y = df4['y_axis']
           ,C = df4['dist'],edgecolors=None,
           reduce_C_function= np.mean,
           gridsize=25,alpha=0.7,vmax=max,vmin=0)


#fig.colorbar(im,cax = cax)
cbar = ax1.cax.colorbar(im)
cbar = axes.cbar_axes[0].colorbar(im)
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)

court = plt.imread("../../Data/court.png")
ax1.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
ax2.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
ax3.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
ax4.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
fig.suptitle('Ball Handler to Closest Defender Distance')

plt.show()
#plt.savefig('DefenceHeatDiff1.png',dpi = 200)

