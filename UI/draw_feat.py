import math
import numpy as np

def distance(x1,y1,x2,y2):
    dist = math.sqrt( ((x2-x1)**2)+ ((y2-y1)**2))
    return dist

def get_feature(real_):
    BASKET_RIGHT = np.array([90.5, 24.8] * 100)
    BASKET_RIGHT = np.reshape(BASKET_RIGHT, newshape=[100, 2])

    en_ball = []
    b_feat = []
    tmp_ball = -1

    data = real_

    len_ = len(real_)

    for i in range(len_):
        tmp = []
        ballx = data[i, 0]
        bally = data[i, 1]

        p1x = data[ i, 2]
        p1y = data[ i, 3]
        p1d = distance(ballx, bally, p1x, p1y)

        p2x = data[i, 4]
        p2y = data[i, 5]
        p2d = distance(ballx, bally, p2x, p2y)

        p3x = data[i, 6]
        p3y = data[i, 7]
        p3d = distance(ballx, bally, p3x, p3y)

        p4x = data[i, 8]
        p4y = data[i, 9]
        p4d = distance(ballx, bally, p4x, p4y)

        p5x = data[i, 10]
        p5y = data[i, 11]
        p5d = distance(ballx, bally, p5x, p5y)

        basketd = distance(ballx, bally, BASKET_RIGHT[i, 0], BASKET_RIGHT[i, 1])

        tmp.append(p1d)
        tmp.append(p2d)
        tmp.append(p3d)
        tmp.append(p4d)
        tmp.append(p5d)
        tmp.append(basketd)

        p = tmp.index(min(tmp))
        has_ball = p
        if p == 5:
            en_ball.append([0, 0, 0, 0, 0,1])
        else:
            if tmp[p] <1:
                if p == 0:
                    en_ball.append([1, 0, 0, 0, 0,0])
                elif p == 1:
                    en_ball.append([0, 1, 0, 0, 0,0])
                elif p == 2:
                    en_ball.append([0, 0, 1, 0, 0,0])
                elif p == 3:
                    en_ball.append([0, 0, 0, 1, 0,0])
                elif p == 4:
                    en_ball.append([0, 0, 0, 0, 1,0])
                elif p == 5:
                    en_ball.append([0, 0, 0, 0, 0,1])
                else:
                    en_ball.append([0, 0, 0, 0, 0,0])

            elif has_ball is not tmp_ball:
                en_ball.append([0, 0, 0, 0, 0,0])
            else:
                en_ball.append([0, 0, 0, 0, 0,1])

    b_feat.append(en_ball)
    en_ball = []

    b_feat = np.array(b_feat)

    return b_feat


'''
data = np.load('./GUI/new/points2.npy')

print(data.shape)

feat = get_feature(data)

print(feat.shape)
print(feat)
np.save('Bhost.npy',feat)
'''
