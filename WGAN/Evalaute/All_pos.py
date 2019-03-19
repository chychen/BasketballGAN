import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

DATA_PATH = './Data/Samples'

real = np.load(DATA_PATH+'/Real/RealTrain.npy')
batch = np.load(DATA_PATH+'/WithPen/Train_Samples.npy')
vae = np.load(DATA_PATH+'/VAE/Train_VAESamples.npy')
gen = np.load(DATA_PATH+'/WithPen/Train_Samples.npy')
#wgan = np.load('./Data/Samples/WGAN/Train_Samples.npy')
wgan = np.load('./Data/Samples/WGANPen/Train_Samples420.npy')

court = plt.imread("../Data/court.png")

sns.set(rc={'lines.linewidth':1},font_scale=0.1)
sns.set_style("darkgrid")
#sns.set_palette("Reds", 100 * length[n])
sns.set_palette("Reds", 1)
'''
for i in range(len(real)):
    p1x = real[i, :, [12, 14, 16, 18, 20]]
    p1y = real[i, :, [13, 15, 17, 19, 21]]
    # Generated data
    b1x = batch[i, :, [12, 14, 16, 18, 20]]
    b1y = batch[i, :, [13, 15, 17, 19, 21]]
    # Condition data
    # s1x = seq[n, :, [12,14,16,18,20]]
    # s1y = seq[n, :,  [13,15,17,19,21]]
    # VAE
'''
b1x = batch[:, :, [12, 14, 16, 18, 20]]
b1y = batch[:, :, [13, 15, 17, 19, 21]]

v1x = vae[:, :, [12, 14, 16, 18, 20]]
v1y = vae[:, :, [13, 15, 17, 19, 21]]

p1x = real[:, :, [12, 14, 16, 18, 20]]
p1y = real[:, :, [13, 15, 17, 19, 21]]

g1x = gen[:, :, [12, 14, 16, 18, 20]]
g1y = gen[:, :, [13, 15, 17, 19, 21]]

w1x = wgan[:,:,[12, 14, 16, 18, 20]]
w1y = wgan[:,:,[13, 15, 17, 19, 21]]

high = ((0., 0., 0.),
         (.03, 0.5, 0.5),
         (1., 1., 1.))

middle = ((0., .2, .2),
           (.05, .5, .3),
           (.8, .2, .2),
           (1., .1, .1))

none = ((0,0,0),
              (1,0,0))

cdict3 = {'red':  none,

     'green': middle,

     'blue': high,

     'alpha': ((0.0, 0.0, 0.0),
               (0.3, 0.5, 0.5),
               (1.0, 1.0, 1.0))
    }

dropout_high = LinearSegmentedColormap('Dropout', cdict3)
plt.register_cmap(cmap = dropout_high)

a = w1x.flatten()
b = w1y.flatten()

#heatmap, xedges, yedges = np.histogram2d(a, b, bins=(47,50))
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.hexbin(a,b,gridsize = 45,cmap=dropout_high,linewidths=0)

#plt.imshow(heatmap.T,extent=extent, origin='lower')
plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
plt.xlim(47, 94)
plt.axis('off')

#plt.show()
plt.savefig('./Image/scatterheatWGANPen.png',dpi=100)

