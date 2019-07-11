import numpy as np
import matplotlib.pyplot as plt
import os
import math
from utils import DataFactory

# data factory
data_path = '/workspace/data/nctu_cgvlab_bballgan/Reordered_Data'
real_data = np.load(os.path.join(data_path, '50Real.npy'))
seq_data = np.load(os.path.join(data_path, '50Seq.npy'))
features_ = np.load(os.path.join(data_path, 'SeqCond.npy'))
real_feat = np.load(os.path.join(data_path, 'RealCond.npy'))

r = np.reshape(real_data[:,:,:6,:2], [-1, 50, 12])

data_factory = DataFactory(
            real_data=real_data,
            seq_data=seq_data,
            features_=features_,
            real_feat=real_feat)

s = data_factory.recover_seq(seq_data)
target = np.stack([r, s], axis=0)
print(target.shape)# shape=[2,?,50,12]

#data path
save_path = '/workspace/data/nctu_cgvlab_bballgan/REALnSEQ/'
for index in range(target.shape[1]):
    num_seg = 2
    length_ = 50
    frames_ = math.ceil(length_ / num_seg)
    print("index:{}/{}".format(index, target.shape[1]))
    start = 0  #starting point
    end = frames_  #ending point (20 frames per image)
    for i in range(num_seg):
        fig = plt.figure(1)
        ax = plt.subplot(1, 1, 1)
        colors = ['r', 'b']
        for t in range(2):
            data = target[t, index]
            #save image file path
            file_name = str(index)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            p2x = np.array(data[:, 2])
            p2y = np.array(data[:, 3])

            court = plt.imread(
                "/workspace/data/nctu_cgvlab_bballgan/Data/court.png")

            total_ = length_ - 1

            if i + 1 == num_seg:
                end = length_
            for x in range(start, end):
                #alpha higher as timestep increases
                alpha_ = 1.0 / (end - start) * (x + 1 - start)
                #offensive player trajectory * 5
                ax.plot(
                    p2x[x:end],
                    p2y[x:end],
                    c=colors[t],
                    alpha=alpha_,
                    linewidth=0,
                    marker='o')

            plt.axis('off')
            plt.xlim(47, 94)
            plt.ylim(50, 0)
            plt.axis("off")
            plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
        #plt.show()
        file_path = save_path + file_name + '_{}.png'.format(i)
        if os.path.isfile(file_path):
            os.remove(file_path)
        plt.savefig(file_path)
        plt.clf()

        start += frames_
        end += frames_

print("Finished")
