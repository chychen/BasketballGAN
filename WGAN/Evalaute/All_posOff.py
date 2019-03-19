import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

DATA_PATH = '../Data/'

real = np.load(DATA_PATH + '/Valid/RealValid.npy')
#Basketball GAN
batch = np.load(DATA_PATH+'/Valid/Valid/Diff/Valid_ReconDiff.npy')
#BasketballGAN2_WGAN
wgan = np.load('../Data/Valid/Valid/Valid_SamplesWgan.npy')
#VAE
vae = np.load(DATA_PATH + '/Valid/Valid_VAESamples.npy')
#condition
seq = np.load('../Data/Valid/Valid_SamplesWgan.npy')


seq = np.load(DATA_PATH+'Valid/Valid_ConditionData.npy')

#wgan = np.load('./Data/Samples/WGANPen/Train_Samples420.npy')

court = plt.imread("../../Data/court.png")


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
dim = 122

#for dim in range(100):
s1x = seq[:100, :, [2,4,6,8,10]]
s1y = seq[:100, :,  [3,5,7,9,11]]

b1x = batch[:, :, [2, 4, 6, 8, 10]]
b1y = batch[:, :, [3, 5, 7, 9, 11]]

v1x = vae[:, :, [2, 4, 6, 8, 10]]
v1y = vae[:, :, [3, 5, 7, 9, 11]]

p1x = real[:, :, [2, 4, 6, 8, 10]]
p1y = real[:, :, [3, 5, 7, 9, 11]]


w1x = wgan[:,:,[2, 4, 6, 8, 10]]
w1y = wgan[:,:,[3, 5, 7, 9, 11]]

high = ((0., 0., 0.),
         (.03, 0.5, 0.5),
         (1., 1., 1.))

middle = ((0., .2, .2),
           (.05, .5, 0.3),
           (.8, .2, .2),
           (1., .1, .1))

none = ((0,0,0),
              (1,0,0))

cdict3 = {'red':  high,

     'green': none,

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

plt.hexbin(a,b,gridsize = 40,cmap=dropout_high,linewidths=0,vmin=0,vmax=2000)
#plt.colorbar()
'''
cdict3 = {'red': none,

     'green': none,

     'blue': high,

     'alpha': ((0.0, 0.0, 0.0),
               (0.3, 0.5, 0.5),
               (1.0, 1.0, 1.0))
    }
dropout_high = LinearSegmentedColormap('Dropout', cdict3)
plt.register_cmap(cmap = dropout_high)

b1x = batch[10, :, [2, 4, 6, 8, 10]]
b1y = batch[10, :, [3, 5, 7, 9, 11]]
p1x = real[10, :, [2, 4, 6, 8, 10]]
p1y = real[10, :, [3, 5, 7, 9, 11]]
a = p1x.flatten()
b = p1y.flatten()

#plt.hexbin(a,b,gridsize = 30,cmap=dropout_high,linewidths=0)
'''
#plt.imshow(heatmap.T,extent=extent, origin='lower')
plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
plt.xlim(47, 94)
plt.axis('off')

plt.show()
#plt.savefig('./heat_solo/scatter_{}.png'.format(dim),dpi=100)
#plt.clf()

