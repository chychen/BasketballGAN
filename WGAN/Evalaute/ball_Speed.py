import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def distance(x1,y1,x2,y2):
    dist = math.sqrt( ((x2-x1)**2)+((y2-y1)**2))
    return dist

DATA_PATH = '../Data/'

'''
real = np.load(DATA_PATH+'/Real/RealTrain.npy')
batch = np.load(DATA_PATH+'/NoPen/Train_Samples.npy')
seq = np.load(DATA_PATH+'/NoPen/Train_ConditionData.npy')
vae = np.load(DATA_PATH+'/VAE/Train_VAESamples.npy')
wgan = np.load('./Data/Samples/WGAN/Train_Samples.npy')
feat_ = np.load('./Data/Samples/Real/RealCondition.npy')
'''

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
seq = np.load('../Data/Valid/Valid_SamplesWgan.npy')
#seq = np.load('../Data/Valid/Valid_ConditionData.npy')

basketb_right = [90, 25]

feat_ = np.load('../Data/Valid/ValidCondition.npy')


genW_ = np.round(batch[:,:,22:27])
featG_ = genW_.astype(int)

featW_ = np.round(wgan[:,:,22:27])
featW_ = featW_.astype(int)

featV_ = np.round(vae[:,:,22:27])
featV_ = featV_.astype(int)

pass_def = [0,0,0,0,0]

speed = []
speed2 = []
speed3 = []
speed4 = []
speed5 = []

for z in range(1):
    for n in range(len(real)):
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


'''
print("Real: ",np.mean(speed))
print("Gen: ",np.mean(speed2))
print("Cond: ",np.mean(speed3))
print("VAE: ",np.mean(speed4))
print("WGAN:",np.mean(speed5))
'''


#print(max_)

y_axis = []
y2_axis = []
y3_axis = []
y4_axis = []
y5_axis = []

count = 0
for i in range(1,51):
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
    percent_ = (count / len(speed4)) * 100
    y5_axis.append(percent_)
    count = 0

x_axis = np.linspace(1,50,50)

sns.set(rc={'lines.linewidth':1.5})
df = pd.DataFrame({'ft/s': x_axis,
                   'Frequency %' : y_axis
                   })

sns.set_style("darkgrid")
sns.set_palette("Set1")

cond_ = sns.lineplot(x = x_axis,y = y2_axis,palette = 'C5',alpha=0)

r = sns.lineplot('ft/s','Frequency %',data=df,palette='C1').set_title("Average Ball velocity")
sample_ = sns.lineplot(x = x_axis,y = y2_axis,palette = 'C3',)
wgan = sns.lineplot(x=x_axis,y=y5_axis,palette='C7')
vae_ = sns.lineplot(x = x_axis,y=y4_axis,palette='C9')



plt.legend(['Condition','Real','BasketballGAN',"BasketballGAN2",'VAE'])


print("Real: ",np.mean(speed))
print("Gen: ",np.mean(speed2))
print("Cond: ",np.mean(speed3))
print("VAE: ",np.mean(speed4))
print("WGAN:",np.mean(speed5))

print("Real: ",np.std(speed))
print("Gen: ",np.std(speed2))
print("Cond: ",np.std(speed3))
print("VAE: ",np.std(speed4))
print("WGAN:",np.std(speed5))


#sns.set_context("poster")
sns.despine(left = True,bottom=True)

plt.show()
#plt.savefig('./Image/PassVelocityPPT.png')
