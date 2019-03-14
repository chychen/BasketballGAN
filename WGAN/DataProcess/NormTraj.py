import numpy as np
from utils import DataFactory
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

#data = np.load('RealPass.npy')
#data = np.load('../Data/Model_data/ShotData/50seq4.npy')
data = np.load('Testseq.npy')
#data = np.load('../Data/Test/TestReal2_D.npy')
tdata = np.load('../Data/Test/TestReal.npy')
print(tdata.shape)
print(data.shape)
point = np.load('50noIdle.npy')
print(point.shape)

'''
minx = 47
maxx = 94
point[:, :, [0,2,4,6,8,10]] = (point[:,:,[0,2,4,6,8,10]] - minx) / (maxx-minx)

miny = 0
maxy = 50
point[:, :, [1,3,5,7,9,11]] = (
                    point[:,:,[1,3,5,7,9,11]] - miny) / (maxy-miny)


plt.plot(point[0,:,2],point[0,:,3])
plt.show()
np.save('50seq4.npy',point)
'''

print(data.shape)

real_data = np.load('../Data/Model_data/F50_D.npy')
print(real_data.shape)
seq_data = np.load('../Data/Model_data/50seq.npy')
features_ = np.load('../Data/Model_data/50Cond.npy')
real_feat = np.load('../Data/Model_data/RealCond.npy')

data_factory = DataFactory(real_data =real_data,seq_data = seq_data, features_= features_,real_feat=real_feat)

'''training_data, valid_data = data_factory.fetch_data()
real_ = training_data['A']
real_def = training_data['B']
real_ = real_[:,:,[0,1,3,4,5,6,7,8,9,10,11,12]]


real_d = np.array(real_[0,0,:])
print('%.16f'%real_d[0])
print(data[0,0,:])
print(point[0,0,:])
plt.plot(data[0,:,2],data[0,:,3])
plt.plot(real_[0,:,2],data[0,:,3])
plt.show()

n_data = data_factory.normalize_back(data)
print(n_data[0,0,:])
plt.plot(n_data[0,:,2],n_data[0,:,3])
plt.show()
'''

#team_A = data[:, :, :, :2].reshape([data.shape[0], data.shape[1], 12])

#print(team_A.shape)

n_data = data_factory.normalize(data)
print(n_data[0,0,:])
print(n_data.shape)
#plt.plot(n_data[0,:,2],n_data[0,:,3])
#plt.show()

np.save('TestSeq.npy',n_data)


