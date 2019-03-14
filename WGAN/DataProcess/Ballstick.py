#Sticks ball to ball handler
#Ball trajectory is same as ball handler trajectory
import numpy as np
import math

def distance(x1,y1,x2,y2):
    dist = math.sqrt( ((x2-x1)**2)+ ((y2-y1)**2))
    return dist

DATA_PATH = '../Data/'
input_file = '../Data/Test/TestReal2_D.npy'
len_file = '../Data/Test/TestLength.npy'
output_file = 'TStick.npy'

data = np.load(input_file)
len_ = np.load(len_file)

BASKET_RIGHT = np.array([88, 25]*235)
BASKET_RIGHT = np.reshape(BASKET_RIGHT,newshape=[235,2])

x = 2

has_ball = 100
dribble = -1
dribbler = []
pass_count = 0
tmp_ball = -1
f_count = 0
get_frame = [0]
new_frame = []
frame = []
final_data = []
count = 0
pass_frame = []

dr = -1

ball_frame = []

for x in range(len(data)):
    for i in range(235):
        tmp = []
        ballx = data[x,i,0, 0]
        bally = data[x,i,0, 1]

        p1x = data[x,i,1,0]
        p1y = data[x,i,1,1]
        p1d = distance(ballx,bally,p1x,p1y)

        p2x = data[x,i,2,0]
        p2y = data[x,i,2,1]
        p2d = distance(ballx,bally,p2x,p2y)

        p3x = data[x,i,3,0]
        p3y = data[x,i,3,1]
        p3d = distance(ballx,bally,p3x,p3y)

        p4x = data[x,i,4,0]
        p4y = data[x,i,4,1]
        p4d = distance(ballx,bally,p4x,p4y)

        p5x = data[x,i,5,0]
        p5y = data[x,i,5,1]
        p5d = distance(ballx,bally,p5x,p5y)

        basketd = distance(ballx,bally,BASKET_RIGHT[i,0],BASKET_RIGHT[i,1])

        tmp.append(p1d)
        tmp.append(p2d)
        tmp.append(p3d)
        tmp.append(p4d)
        tmp.append(p5d)
        tmp.append(basketd)

        p = tmp.index(min(tmp))
        has_ball = p
        if p == 5:
            pass
            data[x, i, 0, 0] = BASKET_RIGHT[0, 0]
            data[x, i, 0, 1] = BASKET_RIGHT[0, 1]
        else:
            f_count += 1
            if tmp[p] < 5:
                if p == 0:
                    data[x, i, 0, 0] = p1x
                    data[x, i, 0, 1] = p1y

                elif p == 1:
                    data[x, i, 0, 0] = p2x
                    data[x, i, 0, 1] = p2y

                elif p == 2:
                    data[x, i, 0, 0] = p3x
                    data[x, i, 0, 1] = p3y

                elif p == 3:
                    data[x, i, 0, 0] = p4x
                    data[x, i, 0, 1] = p4y
                elif p == 4:
                    data[x, i, 0, 0] = p5x
                    data[x, i, 0, 1] = p5y
                elif p == 5:
                    data[x, i, 0, 0] = BASKET_RIGHT[0, 0]
                    data[x, i, 0, 1] = BASKET_RIGHT[0, 1]

                if dribble is not p:
                    dribble = p
                    dribbler.append(p)
                else:
                    pass
                    # bf.append([data[x, i, 0, 0], data[x, i, 0, 1]])

            elif has_ball is not tmp_ball:
                pass
            else:
                pass_frame.append(i)

        if has_ball is not tmp_ball:
            if f_count < 2:
                tmp_ball = has_ball
            else:
                tmp_ball = has_ball
                pass_count += 1
                test = []
                # test.extend(pass_frame)
                test.extend([i] * 2)
                new_frame.append(test)

                # ball_frame.append(bf)
                bf = []

                pass_frame = []
                get_frame.append(i)

            f_count = 0

    ball = data[x,:,0,:2]
    p1 = data[x,:,1,:2]
    p2 = data[x,:,2,:2]
    p3 = data[x,:,3,:2]
    p4 = data[x,:,4,:2]
    p5 = data[x,:,5,:2]

    ndata = ball
    ndata = np.array(ndata)
    ndata = np.append(ndata,p1,axis=1)
    ndata = np.append(ndata,p2,axis=1)
    ndata = np.append(ndata,p3,axis=1)
    ndata = np.append(ndata,p4,axis=1)
    ndata = np.append(ndata,p5,axis=1)

    #final_data.append(ndata)


final_data = np.array(final_data)

np.save(output_file,data)
