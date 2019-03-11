import numpy as np

import os
from ThreeDiscrim import WGAN_Model
from Train_Triple import Training_config
from utils import DataFactory
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DATA_PATH = os.path.join('./Data/')
IMAGE_PATH = os.path.join(DATA_PATH,'images/')
SAMPLE_PATH = os.path.join(DATA_PATH,'400/15/')
MODEL_PATH = os.path.join(DATA_PATH,'checkpoints/model.ckpt-400')
name = "v300_"

class Sampler():
    def __init__(self):
        self.data = None
        self.config = Training_config()
        self.model = WGAN_Model(self.config)
        self.model.load_model(MODEL_PATH)

    def generate_z(self):
        z = np.random.normal(size=(128, self.model.latent_dims)).astype(np.float32)
        return z

    def reconstruct(self, data):
        return self.model.reconstruct_(data)

    def reconstruct_img(self, data,x,feat):
        return self.model.reconstruct_(data,x,feat)

def update_all(frame_id, player_circles, ball_circle, annotations, data, player_circles2, ball_circle2, annotations2, data2,
               player_circles3, ball_circle3, annotations3, data3,cond,cond2,cond3
               ):
    #Ground Truth
    for j, circle in enumerate(player_circles):
        if j < 5:
            cond_ = cond2[frame_id]
            if cond_[j] == 1:
                circle.set_color('yellow')
            elif cond_[j] == 0:
                circle.set_color('r')
        circle.center = data[frame_id, 2+j * 2 + 0], data[frame_id, 2+j * 2 + 1]
        annotations[j].set_position(circle.center)
    # ball
    ball_circle.center = data[frame_id, 0], data[frame_id, 1]

    annotations[10].set_position(ball_circle.center)

    #Sample
    for j, circle in enumerate(player_circles2):
        if j < 5:
            cond_ = cond3[frame_id]
            if cond_[j] == 1:
                circle.set_color('yellow')
            elif cond_[j] == 0:
                circle.set_color('r')
        circle.center = data2[frame_id,  2+j * 2 + 0], data2[frame_id, 2+j * 2 + 1]
        annotations2[j].set_position(circle.center)
    # ball
    ball_circle2.center = data2[frame_id, 0], data2[frame_id, 1]
    annotations2[10].set_position(ball_circle2.center)

    #Condition
    for j, circle in enumerate(player_circles3):
        if j < 5:
            cond_ = cond[frame_id]
            if cond_[j] == 1:
                circle.set_color('yellow')
            elif cond_[j] == 0:
                circle.set_color('r')
        circle.center = data3[frame_id,  2+j * 2 + 0], data3[frame_id, 2+j * 2 + 1]
        annotations3[j].set_position(circle.center)
    # ball
    ball_circle3.center = data3[frame_id, 0], data3[frame_id, 1]

    annotations3[5].set_position(ball_circle3.center)
    return


def plot_data(ax,data, length, fps=6, dpi=300,has_defence = True):
    court = plt.imread("./Data/court.png")  # 500*939
    name_list = ['1', '2', '3', '4', '5',
                 '1','2', '3', '4', '5',
                 '0']

    # team A -> red circle, ball -> small green circle
    player_circles = []
    if has_defence:
        for _ in range(5):
            player_circles.append(plt.Circle(xy=(0, 0), radius=1.75, color='r'))
        for _ in range(5):
            player_circles.append(plt.Circle(xy=(0, 0), radius=1.75, color='b'))

        annotations = [ax.annotate(name_list[i], xy=[47., 0.],
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold',fontsize=5)
                       for i in range(11)]


    else:
        for _ in range(5):
            player_circles.append(plt.Circle(xy=(0, 0), radius=1.75, color='r'))

        annotations = [ax.annotate(name_list[i], xy=[47., 0.],
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold',fontsize=5)
                       for i in range(6)]

    ball_circle = plt.Circle(xy=(0, 0), radius=0.5, color='g')


    for circle in player_circles:
        ax.add_patch(circle)
    ax.add_patch(ball_circle)

    # annotations on circles

    ax.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])

    return player_circles, ball_circle, annotations, data


sampler = Sampler()

#real_data = np.load('../Data/Model_data/F50_D.npy')
real_data = np.load('../Data/Model_data/F50_D.npy')
image_data = np.load('../Data/Model_data/ShotData/50seq2.npy')
features_ = np.load('../Data/Model_data/ShotData/SeqCond.npy')
real_feat = np.load('../Data/Model_data/ShotData/RealCond.npy')

data_factory = DataFactory(real_data,image_data,features_,real_feat)


training_data, valid_data = data_factory.fetch_data()
training_img, valid_img = data_factory.fetch_seq()
f_train,f_valid = data_factory.fetch_feat()
rf_train,rf_valid = data_factory.fetch_realF()

z_sample = np.random.normal(0.,1.,size=[sampler.config.batch_size,sampler.config.latent_dims])

n = 0
batch_id = 15

seq_complete = []
real_complete =[]
sample_complete = []
recon_complete = []

#num_batch = training_data['A'].shape[0] // sampler.config.batch_size
num_batch = valid_data['A'].shape[0] // sampler.config.batch_size

data_idx = batch_id*sampler.config.batch_size%(valid_data['A'].shape[0]-sampler.config.batch_size)
real_data = valid_data['A'][data_idx:data_idx+sampler.config.batch_size]
real_d = valid_data['B'][data_idx:data_idx+sampler.config.batch_size]

data = valid_img[data_idx:data_idx + sampler.config.batch_size]

r_feat = f_valid[data_idx:data_idx + sampler.config.batch_size]
rf_feat = rf_valid[data_idx:data_idx + sampler.config.batch_size]

'''
data_idx = batch_id * sampler.config.batch_size % (training_data['A'].shape[0] - sampler.config.batch_size)
real_data = training_data['A'][data_idx:data_idx + sampler.config.batch_size]
real_d = training_data['B'][data_idx:data_idx + sampler.config.batch_size]

data = training_img[data_idx:data_idx+sampler.config.batch_size]

r_feat = f_train[data_idx:data_idx+sampler.config.batch_size]
rf_feat = rf_train[data_idx:data_idx+sampler.config.batch_size]
'''

real_data = real_data[:,:,[0,1,3,4,5,6,7,8,9,10,11,12]]

seq_data = data

seq = seq_data

truth = real_data
truth_d = real_d
truth = np.concatenate((truth,truth_d),axis=-1)

recon = sampler.reconstruct_img(seq_data,z_sample,r_feat)


sample = data_factory.recover_data(recon[:,:,:22])


truth_ = data_factory.recover_data(truth)
seq_ = data_factory.recover_BALL_and_A(seq)

sample_feat = np.round(recon[:,:,22:])
sample_feat = sample_feat.astype(int)

np.save("Real_traj.npy",truth_)
np.save("Valid_Gen.npy",sample)

for n in range(len(truth)):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_xlim(47, 94)
    ax1.set_ylim(50, 0)

    ax2.set_xlim(47, 94)
    ax2.set_ylim(50, 0)

    ax3.set_xlim(47, 94)
    ax3.set_ylim(50, 0)

    ax1.axis("off")
    ax2.axis("off")
    ax1.set_title("Ground Truth")
    ax2.set_title("Reconstruct")

    ax3.axis("off")
    ax3.set_title("Seq")

    player_circles, ball_circle, annotations, data = plot_data(ax=ax1, data=truth_[n], length=sampler.config.seq_length)

    player_circles3, ball_circle3, annotations3, data3 = plot_data(ax=ax3, data=seq_[n], length=sampler.config.seq_length,
                                                                   has_defence=False)

    player_circles2, ball_circle2, annotations2, data2 = plot_data(ax=ax2, data=sample[n],
                                                                   length=sampler.config.seq_length)
    anim = animation.FuncAnimation(fig, update_all, fargs=(
        player_circles, ball_circle, annotations, data, player_circles2, ball_circle2, annotations2, data2,
        player_circles3, ball_circle3, annotations3, data3,
        r_feat[n], rf_feat[n],sample_feat[n]
    ), frames=sampler.config.seq_length, interval=100)

    anim.save(SAMPLE_PATH + name + "_" + "{}_{}.mp4".format(batch_id, n), fps=6, dpi=300, writer='ffmpeg')
    print("Saved ",n)
    plt.cla()
    plt.clf()

print("Finish")



