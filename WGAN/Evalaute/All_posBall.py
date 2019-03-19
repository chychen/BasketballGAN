import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

DATA_PATH = './Data/Samples'

real = np.load(DATA_PATH+'/Real/RealTrain.npy')
batch = np.load(DATA_PATH+'/NoPen/Train_Samples.npy')
vae = np.load(DATA_PATH+'/VAE/Train_VAESamples.npy')
gen = np.load(DATA_PATH+'/WithPen/Train_Samples.npy')
seq = np.load(DATA_PATH+'/WithPen/Train_ConditionData.npy')
wgan = np.load('./Data/Samples/WGANPen/Train_Samples420.npy')

print(len(gen))
print(len(real))

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

s1x = seq[:, :, 0]
s1y = seq[:, :,  1]

b1x = batch[:, :, 0]
b1y = batch[:, :, 1]

v1x = vae[:, :, 0]
v1y = vae[:, :, 1]

p1x = real[:, :, 0]
p1y = real[:, :, 1]

g1x = gen[:, :, 0]
g1y = gen[:, :, 1]

w1x = wgan[:,:,0]
w1y = wgan[:,:,1]

high = ((0., 0., 0.),
         (.03, .5, .5),
         (1., 1., 1.))

middle = ((0., .2, .2),
           (.5, .03, 0.5),
           (.8, .2, .2),
           (1., .1, .1))

none = ((0,0,0),
              (1,0,0))

cdict3 = {'red':  none,

     'green': high,

     'blue': none,

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

plt.hexbin(a,b,gridsize = 40,cmap=dropout_high,linewidths=0)
#plt.colorbar()

#plt.imshow(heatmap.T,extent=extent, origin='lower')
plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
plt.xlim(47, 94)
plt.axis('off')

#plt.show()
plt.savefig('./Image/scatterheatWGANBallPen.png',dpi=100)

