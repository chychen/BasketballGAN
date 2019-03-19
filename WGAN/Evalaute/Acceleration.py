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

DATA_PATH = '../Data/'

'''
real = np.load(DATA_PATH+'/Real/RealTrain.npy')
batch = np.load(DATA_PATH+'/WithPen/Train_Samples.npy')
seq = np.load(DATA_PATH+'/WithPen/Train_ConditionData.npy')


vae = np.load(DATA_PATH+'/VAE/Train_VAESamples.npy')
wgan = np.load('./Data/Samples/WGAN/Train_Samples.npy')
'''

#Real
real = np.load(DATA_PATH + '/Valid/RealValid.npy')
#Basketball GAN
batch = np.load(DATA_PATH+'/Valid/Valid/Diff/Valid_ReconDiff.npy')
#BasketballGAN2_WGAN
wgan = np.load('../Data/Valid/Valid/Valid_SamplesWgan.npy')
#VAE
vae = np.load(DATA_PATH + '/Valid/Valid_VAESamples.npy')
#condition
seq = np.load('../Data/Valid/Valid_SamplesWgan.npy')
#seq = np.load('../Data/Valid/Valid_ConditionData.npy')

print(wgan.shape)

acceleration = []
acceleration2 = []
acceleration3 = []
acceleration4 = []
acceleration5 = []

posx = 12
posy = 13

for z in range(4):
    for n in range(len(real)):
        speed = []
        speed2 = []
        speed3 = []
        speed4 = []  # VAE
        speed5 = []
        #Real data
        p1x = real[n, :, posx]
        p1y = real[n, :, posy]
        #Generated data
        b1x = batch[n, :, posx]
        b1y = batch[n, :, posy]
        #Condition data
        s1x = seq[n,:,posx]
        s1y = seq[n,:,posy]
        #VAE
        v1x = vae[n,:,posx]
        v1y = vae[n,:,posy]
        #WGAN
        w1x = wgan[n,:,posx]
        w1y = wgan[n,:,posy]

        for x in range(49):
            #calculate distance from one frame to another
            d = distance(p1x[x],p1y[x],p1x[x+1],p1y[x+1])
            d2 = distance(b1x[x], b1y[x], b1x[x + 1], b1y[x + 1])
            d3 = distance(s1x[x], s1y[x], s1x[x + 1], s1y[x + 1])
            d4 = distance(v1x[x],v1y[x],v1x[x+1],v1y[x+1])
            d5 = distance(w1x[x],w1y[x],w1x[x+1],w1y[x+1])

            #Distance to velocity
            dist = (d/0.2)
            dist2 = (d2/0.2)
            dist3 = (d3/0.2)
            dist4 = (d4/0.2)
            dist5 = (d5/0.2)

            #Real
            velocity = np.rint(dist)
            speed.append(velocity)
            #Sample
            velocity = np.rint(dist2)
            speed2.append(velocity)
            #Seq
            velocity = np.rint(dist3)
            speed3.append(velocity)
            #VAE
            velocity = np.rint(dist4)
            speed4.append(velocity)
            #WGAN
            velocity = np.rint(dist5)
            speed5.append(velocity)

        for x in range(48):
            #Real
            acc = abs(speed[x]-speed[x+1])
            acceleration.append(acc)
            #Sample
            acc = abs(speed2[x] - speed2[x + 1])
            acceleration2.append(acc)
            #Seq
            acc = abs(speed3[x] - speed3[x + 1])
            acceleration3.append(acc)
            #VAE
            acc = abs(speed4[x] - speed4[x + 1])
            acceleration4.append(acc)
            #WGAN
            acc = abs(speed5[x]-speed5[x+1])
            acceleration5.append(acc)

    posx+=2
    posy+=2

print("Real: ",np.mean(acceleration))
print(np.std(acceleration))
print("Condition: ",np.mean(acceleration3))
print(np.std(acceleration3))
print("BasketballGAN: ",np.mean(acceleration2))
print(np.std(acceleration2))
print("Wgan:",np.mean(acceleration5))
print(np.std(acceleration5))
print("VAE: ",np.mean(acceleration4))
print(np.std(acceleration4))


y_axis = []
y2_axis = []
y3_axis = []
y4_axis = []
y5_axis = []

count = 0
for i in range(9):
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

x_axis = np.linspace(0,8,9)

'''
plt.plot(x_axis,y_axis,c='b')
plt.plot(x_axis,y2_axis,c='g')
plt.plot(x_axis,y3_axis,c='r')
'''

sns.set(rc={'lines.linewidth':1.5})

sns.set_style("darkgrid")
sns.set_palette("Set1")


my_colors = ['b','g','purple','orange']

df = pd.DataFrame({#'Data0':y3_axis,
                    'Data1':y_axis,
                   'Data2':y2_axis,
                   'Data3':y5_axis,
                   'Data6':y4_axis}).plot(kind='line',color= my_colors)

plt.xticks(x_axis)
plt.title('Defensive Mean Acceleration')
plt.xlabel('Ft/s^2')
plt.ylabel('Probability')

plt.legend(['Real','BasketballGAN',"BasketballGAN_2",'C-VAE'])

sns.despine(left = True,bottom=True)

#plt.show()
plt.savefig('../../../Graphs/DefenceAcceleration.svg')
