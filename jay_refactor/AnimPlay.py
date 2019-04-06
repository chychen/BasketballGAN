import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import shutil

def update_all(frame_id, player_circles, ball_circle, annotations, data):
    """
    Inputs
    ------
    frame_id : int
        automatically increased by 1
    player_circles : list of pyplot.Circle
        players' icon
    ball_circle : list of pyplot.Circle
        ball's icon
    annotations : pyplot.axes.annotate
        colors, texts, locations for ball and players
    data : float, shape=[amount, length, 23]
        23 = ball's xyz + 10 players's xy
    """
    # players

    for j, circle in enumerate(player_circles):
        #circle.center = data[frame_id,2 + j * 2 + 0], data[frame_id, 2 + j * 2 + 1]
        circle.center = data[frame_id, 2+j * 2 + 0], data[frame_id, 2+j * 2 + 1]
        annotations[j].set_position(circle.center)
    # ball
    ball_circle.center = data[frame_id, 0], data[frame_id, 1]

    # players

    return


def plot_data(ax,data, length, fps=6, dpi=300):
    """
    Inputs
    ------
    data : float, shape=[amount, length, 23]
        23 = ball's xyz + 10 players's xy
    length : int
        how long would you like to plot
    file_path : str
        where to save the animation
    if_save : bool, optional
        save as .gif file or not
    fps : int, optional
        frame per second
    dpi : int, optional
        dot per inch
    Return
    ------
    """
    court = plt.imread("/workspace/data/nctu_cgvlab_bballgan/Data/court.png")  # 500*939
    name_list = ['1', '2', '3', '4', '5',
                 '1','2','3','4','5']

    # team A -> red circle, ball -> small green circle
    player_circles = []
    [player_circles.append(plt.Circle(xy=(0, 0), radius=2, color='r',alpha = 0.75))
     for _ in range(5)]
    [player_circles.append(plt.Circle(xy=(0, 0), radius=2, color='b',alpha = 0.75))
     for _ in range(5)]
    ball_circle = plt.Circle(xy=(0, 0), radius=1, color='g')

    # plot
    #ax = plt.axes(xlim=(0, 100), ylim=(0, 50))
    #ax.axis('off')

    #ax.grid(False)
    for circle in player_circles:
        ax.add_patch(circle)
    ax.add_patch(ball_circle)

    # annotations on circles
    annotations = [ax.annotate(name_list[i], xy=[47., 0.],
                               horizontalalignment='center',
                               verticalalignment='center', fontweight='bold')
                   for i in range(10)]
    # animation
    '''
    anim = animation.FuncAnimation(fig, update_all, fargs=(
        player_circles, ball_circle, annotations, data), frames=length, interval=100)
        '''


    ax.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])

    return player_circles, ball_circle, annotations, data

#DATA_PATH = './Data/Real/'
#data = np.load('PaddedPoint.npy')

DATA_PATH = '/workspace/data/nctu_cgvlab_bballgan/Log/acc_pen/'
data = np.load(DATA_PATH+'diff_len_results/reconstruct.npy')
len_ = np.load('/workspace/data/nctu_cgvlab_bballgan/Reordered_Data/Test2/TestLength2.npy')
# data = np.load('/workspace/data/nctu_cgvlab_bballgan/Log/CVAE/reconstruct.npy')
len_ = np.ones(shape=data.shape[0], dtype=np.int32)*50

print(data.shape)

#n = 512

save_path = DATA_PATH+'diff_len_results/Videos'
# save_path = '/workspace/data/nctu_cgvlab_bballgan/Log/CVAE/Videos'
if os.path.exists(save_path):
    shutil.rmtree(save_path)
    print('rm -rf "%s" complete!' % save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)
for n in range(1000):
    ndata = data[n]

    # print(ndata.shape)
    '''
    if x == 0:
        final_data = ndata
    else:
        dataa = np.column_stack(final_data,ndata)
        final_data = np.append(final_data,ndata,axis = 1)
    '''

    truth = ndata

    fig, ax1 = plt.subplots(1,1)
    ax1.set_xlim(47, 94)
    ax1.set_ylim(50, 0)
    ax1.axis("off")

    player_circles, ball_circle, annotations, datax = plot_data(ax=ax1, data=truth, length=len_[n])


    anim = animation.FuncAnimation(fig, update_all, fargs=(
        player_circles, ball_circle, annotations, datax),
                                   frames=len_[n], interval=150)
    anim.save(save_path+'/GEN{}.mp4'.format(n), fps=5, dpi=300, writer='ffmpeg')

    plt.clf()
    #plt.show()