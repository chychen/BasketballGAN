import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def distance(x1,y1,x2,y2):
    dist = math.sqrt( ((x2-x1)**2)+((y2-y1)**2))
    return dist

DATA_PATH = '../Data/'
#Real
real = np.load(DATA_PATH + '/Valid/RealValid.npy')
#Basketball GAN
batch = np.load(DATA_PATH+'/Valid/Valid/Diff/Valid_ReconDiff.npy')
#batch = np.load(DATA_PATH+'/Final/Valid_ReconDiff_v2.npy')
#BasketballGAN2_WGAN
wgan = np.load('../Data/Valid/Valid/Valid_SamplesWgan.npy')
#VAE
vae = np.load(DATA_PATH + '/Valid/Valid_VAESamples.npy')
#condition
#seq = np.load('../Data/Valid/Valid_SamplesWgan.npy')
seq = np.load('../Data/Valid/Valid_ConditionData.npy')

basketb_right = [90, 25]

feat_ = np.load('../Data/Valid/ValidCondition.npy')

speed = []  #Real
speed2 = [] #Sample
speed3 = [] #SEQ
speed4 = [] #VAE
speed5 = []
speed6 = [] #Diff
speed7 = [] #Global

posx = 2
posy = 3

for z in range(4):
    for n in range(len(real)):
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

    posx+=2
    posy+=2


print("Real: ",np.mean(speed))
print(np.std(speed))
print("Condition: ",np.mean(speed3))
print(np.std(speed3))
print("BasketballGAN: ",np.mean(speed2))
print(np.std(speed2))
print("Wgan:",np.mean(speed5))
print(np.std(speed5))
print("VAE: ",np.mean(speed4))
print(np.std(speed4))

min_ = min(speed)
max_ = max(speed)
#print(max_)

y_axis = []
y2_axis = []
y3_axis = []
y4_axis = []
y5_axis = []


count = 0
for i in range(21):
    for x in speed:
        if x == i:
            count += 1

    percent_ = (count/len(speed))*100
    y_axis.append(percent_)
    count = 0

    for x in speed2:
        if x == i:
            count += 1

    percent_ = (count/len(speed2))*100
    y2_axis.append(percent_)
    count = 0

    for x in speed3:
        if x == i:
            count += 1
    percent_ = (count / len(speed3)) * 100
    y3_axis.append(percent_)
    count = 0

    for x in speed4:
        if x == i:
            count += 1
    percent_ = (count / len(speed4)) * 100
    y4_axis.append(percent_)
    count = 0

    for x in speed5:
        if x == i:
            count += 1
    percent_ = (count / len(speed5)) * 100
    y5_axis.append(percent_)
    count = 0


x_axis = np.linspace(0,20,21)

sns.set(rc={'lines.linewidth':1.5})

sns.set_style("darkgrid")
sns.set_palette("Set1")


my_colors = ['r','b','g','purple','orange']

df = pd.DataFrame({'Data0':y3_axis,
                   'Data1':y_axis,
                   'Data2':y2_axis,
                   'Data3':y5_axis,
                   'Data6':y4_axis}).plot(kind='line',color= my_colors)

plt.xticks(x_axis)
plt.title('Offensive Mean Velocity')
plt.xlabel('Ft/s')
plt.ylabel('Probability')

plt.legend(['Condition','Real','BasketballGAN',"BasketballGAN_2",'C-VAE'])

sns.despine(left = True,bottom=True)

#plt.show()
plt.savefig('../../../Graphs/OffenceVelocityValid.svg')
