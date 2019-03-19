import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

'''
court is 94feet long
50feet wide
ft/s
[0,1] = ball
[2,3] = p1
[4,5] = p2
[6,7] = p3
[8,9] = p4
[10,11] = p5
'''

def distance(x1,y1,x2,y2):
    dist = math.sqrt( ((x2-x1)**2)+((y2-y1)**2))
    return dist

DATA_PATH = './Data/Samples'


real = np.load(DATA_PATH+'/Real/RealTrain.npy')
batch = np.load(DATA_PATH+'/NoPen/Train_Samples.npy')
seq = np.load(DATA_PATH+'/WithPen/Train_ConditionData.npy')
vae = np.load(DATA_PATH+'/VAE/Train_VAESamples.npy')
wgan = np.load('./Data/Samples/WGAN/Train_Samples.npy')

feat_ = np.load('./Data/Samples/Real/RealCondition.npy')

genW_ = np.round(batch[:,:,22:])
featG_ = genW_.astype(int)

featW_ = np.round(wgan[:,:,22:])
featW_ = featW_.astype(int)

featV_ = np.round(vae[:,:,22:])
featV_ = featV_.astype(int)

pass_def = [0,0,0,0,0]


acceleration = []
acceleration2 = []
acceleration3 = []
acceleration4 = []
acceleration5 = []

for z in range(1):
    for n in range(len(real)):
        speed = []
        speed2 = []
        speed3 = []
        speed4 = []  # VAE
        speed5 = []

        test = feat_[n]
        testC = feat_[n]

        testG = featG_[n]
        testV = featV_[n]
        testW = featW_[n]

        #Real data
        p1x = real[n, :, 0]
        p1y = real[n, :, 1]
        #Generated data
        b1x = batch[n, :, 0]
        b1y = batch[n, :, 1]
        #Condition data
        s1x = seq[n,:,0]
        s1y = seq[n,:,1]
        #VAE
        v1x = vae[n,:,0]
        v1y = vae[n,:,1]
        #WGAN
        w1x = wgan[n,:,0]
        w1y = wgan[n,:,1]


        for x in range(49):
            # calculate distance from one frame to another
            if (test[x] == pass_def).all():
                d = distance(p1x[x-1],p1y[x-1],p1x[x],p1y[x])
                dist = (d / 0.2)
                velocity = np.rint(dist)
                speed.append(velocity)
                if not (test[x+1] == pass_def).all():
                    d = distance(p1x[x], p1y[x], p1x[x+1], p1y[x+1])
                    dist = (d / 0.2)
                    velocity = np.rint(dist)
                    speed.append(velocity)
                    for i in range(len(speed)-1):
                        acc = abs(speed[i] - speed[i + 1])
                        acceleration.append(acc)
                    speed = []

            if (testC[x] == pass_def).all():
                d = distance(s1x[x-1],s1y[x-1],s1x[x],s1y[x])
                dist = (d / 0.2)
                velocity = np.rint(dist)
                speed3.append(velocity)
                if not (testC[x+1] == pass_def).all():
                    d = distance(s1x[x], s1y[x], s1x[x+1], s1y[x+1])
                    dist = (d / 0.2)
                    velocity = np.rint(dist)
                    speed3.append(velocity)
                    for i in range(len(speed3)-1):
                        acc = abs(speed3[i] - speed3[i + 1])
                        acceleration3.append(acc)
                    speed3 = []

            if (testG[x] == pass_def).all():
                d = distance(b1x[x-1],b1y[x-1],b1x[x],b1y[x])
                dist = (d / 0.2)
                velocity = np.rint(dist)
                speed2.append(velocity)
                if not (testG[x+1] == pass_def).all():
                    d = distance(b1x[x], b1y[x], b1x[x+1], b1y[x+1])
                    dist = (d / 0.2)
                    velocity = np.rint(dist)
                    speed2.append(velocity)
                    for i in range(len(speed2)-1):
                        acc = abs(speed2[i] - speed2[i + 1])
                        acceleration2.append(acc)
                    speed2 = []

            if (testW[x] == pass_def).all():
                d = distance(w1x[x-1],w1y[x-1],w1x[x],w1y[x])
                dist = (d / 0.2)
                velocity = np.rint(dist)
                speed5.append(velocity)
                if not (testW[x+1] == pass_def).all():
                    d = distance(w1x[x], w1y[x], w1x[x+1], w1y[x+1])
                    dist = (d / 0.2)
                    velocity = np.rint(dist)
                    speed5.append(velocity)
                    for i in range(len(speed5)-1):
                        acc = abs(speed5[i] - speed5[i + 1])
                        acceleration5.append(acc)
                    speed5 = []

            if (testV[x] == pass_def).all():
                d = distance(v1x[x-1],v1y[x-1],v1x[x],v1y[x])
                dist = (d / 0.2)
                velocity = np.rint(dist)
                speed4.append(velocity)
                if not (testV[x+1] == pass_def).all():
                    d = distance(v1x[x], v1y[x], v1x[x+1], v1y[x+1])
                    dist = (d / 0.2)
                    velocity = np.rint(dist)
                    speed4.append(velocity)
                    for i in range(len(speed4)-1):
                        acc = abs(speed4[i] - speed4[i + 1])
                        acceleration4.append(acc)
                    speed4 = []

y_axis = []
y2_axis = []
y3_axis = []
y4_axis = []
y5_axis = []

count = 0
for i in range(21):
    for x in acceleration:
        if x == i:
            count += 1

    percent_ = (count/len(acceleration))*100
    y_axis.append(percent_)
    count = 0

    for x in acceleration2:
        if x == i:
            count += 1

    percent_ = (count/len(acceleration2))*100
    y2_axis.append(percent_)
    count = 0

    for x in acceleration3:
        if x == i:
            count += 1


    percent_ = (count / len(acceleration3)) * 100
    y3_axis.append(percent_)
    count = 0

    for x in acceleration4:
        if x == i:
            count += 1

    percent_ = (count / len(acceleration4)) * 100
    y4_axis.append(percent_)
    count = 0

    for x in acceleration5:
        if x == i:
            count += 1

    percent_ = (count / len(acceleration5)) * 100
    y5_axis.append(percent_)
    count = 0

x_axis = np.linspace(0,20,21)

'''
plt.plot(x_axis,y_axis,c='b')
plt.plot(x_axis,y2_axis,c='g')
plt.plot(x_axis,y3_axis,c='r')
'''
df = pd.DataFrame({'ft/s^2': x_axis,
                   'Frequency %' : y_axis
                   })

sns.set_style("darkgrid")
sns.set_palette("Set1")

cond_ = sns.lineplot(x = x_axis,y = y3_axis,palette = 'C5').set_title("Average Ball Acceleration")
r = sns.lineplot('ft/s^2','Frequency %',data=df,palette='C1')

sample_ = sns.lineplot(x = x_axis,y = y2_axis,palette = 'C3',)
vae_ = sns.lineplot(x = x_axis,y=y4_axis,palette='C9')
wgan_ = sns.lineplot(x = x_axis,y=y5_axis,palette='C7')


sns.despine(left = True,bottom=True)

plt.legend(['Condition','Real','Generated','VAE','WGAN'])
plt.show()
#plt.savefig('./Image/PassAcceleration.png')