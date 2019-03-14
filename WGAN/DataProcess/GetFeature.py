
import numpy as np
import math


def distance(x1,y1,x2,y2):
    dist = math.sqrt( ((x2-x1)**2)+ ((y2-y1)**2))
    return dist

input_file = "TestSeq.npy"
output_file = "SeqCond.npy"

data = np.load(input_file)

BASKET_RIGHT = np.array([88, 25]*235)
BASKET_RIGHT = np.reshape(BASKET_RIGHT,newshape=[235,2])
print(data.shape)
feature_ = []
datax = []

for x in range(len(data)):
    for i in range(235):
        tmp = []
        ballx = data[x,i,0]
        bally = data[x,i,1]

        p1x = data[x,i,2]
        p1y = data[x,i,3]
        p1d = distance(ballx,bally,p1x,p1y)

        p2x = data[x,i,4]
        p2y = data[x,i,5]
        p2d = distance(ballx,bally,p2x,p2y)

        p3x = data[x,i,6]
        p3y = data[x,i,7]
        p3d = distance(ballx,bally,p3x,p3y)

        p4x = data[x,i,8]
        p4y = data[x,i,9]
        p4d = distance(ballx,bally,p4x,p4y)

        p5x = data[x,i,10]
        p5y = data[x,i,11]
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
        #print(tmp[p])


        if tmp[p] == 0:
            if p == 0:
                feature_.append([1,0,0,0,0,0])
            elif p == 1:
                feature_.append([0, 1, 0, 0, 0,0])
            elif p == 2:
                feature_.append([0, 0, 1, 0, 0,0])
            elif p == 3:
                feature_.append([0, 0, 0, 1, 0,0])
            elif p == 4:
                feature_.append([0, 0, 0, 0, 1,0])
            elif p == 5:
                feature_.append([0, 0, 0, 0, 0,1])

        else:
            feature_.append([0, 0, 0, 0, 0,0])

    #feature_ = np.array(feature_)

    #if 235-len_[x] > 0:
    #    diff_ = 235-len_[x]
    #    zero_ = np.zeros([235,5])
    #    feature_ = np.vstack((feature_,zero_[:diff_]))

    datax.append(feature_)
    feature_ = []
datax = np.array(datax)
print(datax.shape)

np.save(output_file,datax)
print("Saved")



