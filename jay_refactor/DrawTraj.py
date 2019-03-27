import numpy as np
import matplotlib.pyplot as plt
import os
import math

#data path
# root = '/workspace/data/nctu_cgvlab_bballgan/Log/v13_stop_grad/'
# root = '/workspace/data/nctu_cgvlab_bballgan/Log/v12_SN_WGAN_GP/'
reconstruct = []
reconstruct.append(
    np.load(
        '/workspace/data/nctu_cgvlab_bballgan/Log/v13_stop_grad/results/reconstruct.npy'
    ))
reconstruct.append(
    np.load(
        '/workspace/data/nctu_cgvlab_bballgan/Log/v12_SN_WGAN_GP/results/reconstruct.npy'
    ))
num_data = reconstruct[0].shape[0]
titles = ['With Penalty', 'Without Penalty']
for index in range(num_data):
    num_seg = 5
    length_ = 50
    frames_ = math.ceil(length_ / num_seg)
    print("index:{}/{}".format(index, num_data))
    start = 0  #starting point
    end = frames_  #ending point (20 frames per image)
    for i in range(num_seg):
        fig = plt.figure(1)
        for j in range(2):
            ax = plt.subplot(1, 2, j + 1)
            ax.set_title(titles[j])
            data = reconstruct[j][index]
            data[:, [0, 2, 4, 6, 8, 10]] = [
                x for x in data[:, [0, 2, 4, 6, 8, 10]]
            ]

            #save image file path
            save_path = '/workspace/data/nctu_cgvlab_bballgan/Log/Compare_Traj/'
            file_name = str(index)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            p1x = data[:, 0]
            p1y = data[:, 1]
            p2x = data[:, 2]
            p2y = data[:, 3]
            p3x = data[:, 4]
            p3y = data[:, 5]
            p4x = data[:, 6]
            p4y = data[:, 7]
            p5x = data[:, 8]
            p5y = data[:, 9]
            p6x = data[:, 10]
            p6y = data[:, 11]
            p7x = data[:, 12]
            p7y = data[:, 13]
            p8x = data[:, 14]
            p8y = data[:, 15]
            p9x = data[:, 16]
            p9y = data[:, 17]
            p10x = data[:, 18]
            p10y = data[:, 19]
            p11x = data[:, 20]
            p11y = data[:, 21]

            court = plt.imread(
                "/workspace/data/nctu_cgvlab_bballgan/Data/court.png")

            total_ = length_ - 1

            if i + 1 == num_seg:
                end = length_
            for x in range(start, end):
                #alpha higher as timestep increases
                alpha_ = 1.0 / (end - start) * (x + 1 - start)
                #ball trajectory
                ax.plot(
                    p1x[x:end],
                    p1y[x:end],
                    c='g',
                    alpha=alpha_,
                    linewidth=5,
                    solid_capstyle='round')
                #offensive player trajectory * 5
                ax.plot(
                    p2x[x:end],
                    p2y[x:end],
                    c='r',
                    alpha=alpha_,
                    linewidth=5,
                    solid_capstyle='round')
                ax.plot(
                    p3x[x:end],
                    p3y[x:end],
                    c='r',
                    alpha=alpha_,
                    linewidth=5,
                    solid_capstyle='round')
                ax.plot(
                    p4x[x:end],
                    p4y[x:end],
                    c='r',
                    alpha=alpha_,
                    linewidth=5,
                    solid_capstyle='round')
                ax.plot(
                    p5x[x:end],
                    p5y[x:end],
                    c='r',
                    alpha=alpha_,
                    linewidth=5,
                    solid_capstyle='round')
                ax.plot(
                    p6x[x:end],
                    p6y[x:end],
                    c='r',
                    alpha=alpha_,
                    linewidth=5,
                    solid_capstyle='round')

                #defensive player trajecotry * 5
                ax.plot(
                    p7x[x:end],
                    p7y[x:end],
                    c='b',
                    alpha=alpha_,
                    linewidth=4,
                    solid_capstyle='round')
                ax.plot(
                    p8x[x:end],
                    p8y[x:end],
                    c='b',
                    alpha=alpha_,
                    linewidth=4,
                    solid_capstyle='round')
                ax.plot(
                    p9x[x:end],
                    p9y[x:end],
                    c='b',
                    alpha=alpha_,
                    linewidth=4,
                    solid_capstyle='round')
                ax.plot(
                    p10x[x:end],
                    p10y[x:end],
                    c='b',
                    alpha=alpha_,
                    linewidth=4,
                    solid_capstyle='round')
                ax.plot(
                    p11x[x:end],
                    p11y[x:end],
                    c='b',
                    alpha=alpha_,
                    linewidth=4,
                    solid_capstyle='round')

            #trajectory label
            if i + 1 == num_seg:
                ax.annotate('A5', (p2x[total_] - 1, p2y[total_] + 1))
                ax.annotate('A4', (p3x[total_], p3y[total_]))
                ax.annotate('A3', (p4x[total_], p4y[total_] - 0.5))
                ax.annotate('A2', (p5x[total_], p5y[total_] - 1))
                ax.annotate('A1', (p6x[total_], p6y[total_]))
                ax.annotate('B5', (p7x[total_] + 1, p7y[total_] + 1))
                ax.annotate('B4', (p8x[total_], p8y[total_]))
                ax.annotate('B3', (p9x[total_], p9y[total_]))
                ax.annotate('B2', (p10x[total_], p10y[total_]))
                ax.annotate('B1', (p11x[total_], p11y[total_]))
            else:
                ax.annotate('A5', (p2x[end], p2y[end]))
                ax.annotate('A4', (p3x[end], p3y[end]))
                ax.annotate('A3', (p4x[end], p4y[end]))
                ax.annotate('A2', (p5x[end], p5y[end]))
                ax.annotate('A1', (p6x[end], p6y[end]))
                ax.annotate('B5', (p7x[end], p7y[end]))
                ax.annotate('B4', (p8x[end], p8y[end]))
                ax.annotate('B3', (p9x[end], p9y[end]))
                ax.annotate('B2', (p10x[end], p10y[end]))
                ax.annotate('B1', (p11x[end], p11y[end]))

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
