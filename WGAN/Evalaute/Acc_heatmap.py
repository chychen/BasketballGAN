import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1 import AxesGrid

BASKET_RIGHT = [90, 25]


def distance(x1, y1, x2, y2):
    dist = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    return dist


def nearest_defender(player, teamB):
    dist_ = []

    for x in range(5):
        defender_ = teamB[x]
        dist = distance(player[0], player[1], defender_[0], defender_[1])
        dist_.append(dist)
    dist_min = np.min(dist_)

    return dist_min


DATA_PATH = '../Data/'

'''
real = np.load(DATA_PATH+'/Real/RealTrain.npy')
batch = np.load(DATA_PATH+'/WithPen/Train_Samples.npy')
no_pen = np.load(DATA_PATH+'/NoPen/Train_Samples.npy')
vae = np.load(DATA_PATH+'/VAE/Train_VAESamples.npy')

wgan = np.load('./Data/Samples/WGAN/Train_Samples.npy')
basketb_right = [90, 25]


feat_ = np.load('./Data/Samples/Real/RealCondition.npy')
'''

real = np.load(DATA_PATH + '/Valid/RealValid.npy')
#Basketball GAN
batch = np.load(DATA_PATH+'/Valid/Valid/Diff/Valid_ReconDiff.npy')
#batch = np.load(DATA_PATH+'/Final/Valid_ReconDiff_v2.npy')
#BasketballGAN2_WGAN
wgan = np.load('../Data/Valid/Valid/Valid_SamplesWgan.npy')
#VAE
vae = np.load(DATA_PATH + '/Valid/Valid_VAESamples.npy')
#condition
seq = np.load('../Data/Valid/Valid_SamplesWgan.npy')


no_pen = np.load(DATA_PATH + '/Valid/Valid_Gen.npy')


basketb_right = [90, 25]

feat_ = np.load('../Data/Valid/ValidCondition.npy')

pass_def = [0, 0, 0, 0, 0]
pass_ = 0

total_distance = 0

dist_sample = []
distGen_sample = []
distVae_sample = []
distWgan_sample = []
distPen_sample = []

posx = 2
posy = 3

for z in range(4):
    for n in range(len(real)):
        speed = []
        speed2 = []
        speed3 = []
        speed4 = []  # VAE
        speed5 = []
        #Real data
        p1x = real[n, :, posx]
        p1y = real[n, :, posy]
        #Generated data
        b1x = batch[n, :, posx]
        b1y = batch[n, :, posy]
        #VAE
        v1x = vae[n,:,posx]
        v1y = vae[n,:,posy]
        #WGAN
        w1x = wgan[n,:,posx]
        w1y = wgan[n,:,posy]

        for x in range(49):
            #calculate distance from one frame to another
            d = distance(p1x[x],p1y[x],p1x[x+1],p1y[x+1])
            d2 = distance(b1x[x], b1y[x], b1x[x + 1], b1y[x + 1])
            #d3 = distance(s1x[x], s1y[x], s1x[x + 1], s1y[x + 1])
            d4 = distance(v1x[x],v1y[x],v1x[x+1],v1y[x+1])
            d5 = distance(w1x[x],w1y[x],w1x[x+1],w1y[x+1])

            #Distance to velocity
            dist = (d/0.2)
            dist2 = (d2/0.2)
            #dist3 = (d3/0.2)
            dist4 = (d4/0.2)
            dist5 = (d5/0.2)

            #Real
            velocity = np.rint(dist)
            speed.append(velocity)
            #Sample
            velocity = np.rint(dist2)
            speed2.append(velocity)
            #Seq
            #velocity = np.rint(dist3)
            #speed3.append(velocity)
            #VAE
            velocity = np.rint(dist4)
            speed4.append(velocity)
            #WGAN
            velocity = np.rint(dist5)
            speed5.append(velocity)

        for x in range(48):
            # Real
            acc = abs(speed[x] - speed[x + 1])
            # Sample
            acc2 = abs(speed2[x] - speed2[x + 1])
            # Seq
            #acc = abs(speed3[x] - speed3[x + 1])
            #acceleration3.append(acc)
            # VAE
            acc4 = abs(speed4[x] - speed4[x + 1])
            # WGAN
            acc5 = abs(speed5[x] - speed5[x + 1])

            # Real
            dist_sample.append([p1x[x+1], p1y[x+1], np.floor(acc)])
            # Sample
            distGen_sample.append([b1x[x+1], b1y[x+1], np.floor(acc2)])
            # VAE
            distVae_sample.append([v1x[x+1], v1y[x+1], np.floor(acc4)])
            # WGAN
            distWgan_sample.append([w1x[x+1], w1y[x+1], np.floor(acc5)])

    posx+=2
    posy+=2

r_ = np.array(dist_sample)
wgan_ = np.array(distWgan_sample)
gen_ = np.array(distGen_sample)
vae_ = np.array(distVae_sample)

fig = plt.figure(figsize=(10, 6))

axes = AxesGrid(fig, 111,
                nrows_ncols=(2, 2),
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
ax2.set_ylim(50, 0)
ax2.axis("off")
ax2.set_title('BasketballGAN2')

ax3 = axes[1]
ax3.set_xlim(47, 94)
ax3.set_ylim(50, 0)
ax3.axis("off")
ax3.set_title('BasketballGAN')

ax4 = axes[3]
ax4.set_xlim(47, 94)
ax4.set_ylim(50, 0)
ax4.axis("off")
ax4.set_title('C-VAE')

df = pd.DataFrame({'x_axis': r_[:, 0],
                   'y_axis': r_[:, 1],
                   'dist': r_[:, 2]
                   })

df2 = pd.DataFrame({'x_axis': gen_[:, 0],
                    'y_axis': gen_[:, 1],
                    'dist': gen_[:, 2]
                    })
df3 = pd.DataFrame({'x_axis': wgan_[:, 0],
                    'y_axis': wgan_[:, 1],
                    'dist': wgan_[:, 2]
                    })
df4 = pd.DataFrame({'x_axis': vae_[:, 0],
                    'y_axis': vae_[:, 1],
                    'dist': vae_[:, 2]
                    })

max = 5

im = plt.imshow(np.random.random((0, max)), vmin=0, vmax=max, aspect="auto")

ax1.hexbin(x=df['x_axis'], y=df['y_axis']
           , C=df['dist'],
           reduce_C_function=np.mean,edgecolors=None,
           gridsize=30, alpha=0.7, vmax=max, vmin=0)

ax3.hexbin(x=df2['x_axis'], y=df2['y_axis']
           , C=df2['dist'],
           reduce_C_function=np.mean,edgecolors=None,
           gridsize=35, alpha=0.7, vmax=max, vmin=0)

ax2.hexbin(x=df3['x_axis'], y=df3['y_axis']
           , C=df3['dist'],
           reduce_C_function=np.mean,edgecolors=None,
           gridsize=35, alpha=0.7, vmax=max, vmin=0)

ax4.hexbin(x=df4['x_axis'], y=df4['y_axis']
           , C=df4['dist'],
           reduce_C_function=np.mean,edgecolors=None,
           gridsize=35, alpha=0.7, vmax=max, vmin=0)

# fig.colorbar(im,cax = cax)
cbar = ax1.cax.colorbar(im)
cbar = axes.cbar_axes[0].colorbar(im)
# cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)

court = plt.imread("../../Data/court.png")
ax1.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
ax2.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
ax3.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
ax4.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])

fig.suptitle('Offensive Mean Acceleration')

plt.show()
#plt.savefig('speed2.png', dpi=200)

