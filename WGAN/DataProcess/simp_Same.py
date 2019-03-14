import numpy as np
import math
import GenerateTraj as G

#input is ball stick
data_path = '../Data/'
output_file = 'Seq.npy'
#data = np.load(data_path+'FixedFPS5.npy')
len_ = np.load('../Data/Test/TestLength.npy')
data = np.load('../Data/Test/TestReal.npy')

BASKET_RIGHT = np.array([88, 25] * 235)
BASKET_RIGHT = np.reshape(BASKET_RIGHT, newshape=[235, 2])

def distance(x1, y1, x2, y2):
    dist = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    return dist

has_ball = -1
tmp_ball = 10
tmp_i = 0
new_d = []
feature_ = []
speed = 27.65
save_ball = []

for x in range(len(data)):
    print(x)
    count = 0
    pass_ = []
    pass_len = []
    tmp_i = 0
    p_count = 2
    tmp_c = 0
    for i in range(len_[x]):
        tmp = []
        ballx = data[x, i, 0,0]
        bally = data[x, i, 0,1]

        p1x = data[x, i, 1,0]
        p1y = data[x, i, 1,1]
        p1d = distance(ballx, bally, p1x, p1y)

        p2x = data[x, i, 2,0]
        p2y = data[x, i, 2,1]
        p2d = distance(ballx, bally, p2x, p2y)

        p3x = data[x, i, 3,0]
        p3y = data[x, i, 3,1]
        p3d = distance(ballx, bally, p3x, p3y)

        p4x = data[x, i, 4, 0]
        p4y = data[x, i, 4, 1]
        p4d = distance(ballx, bally, p4x, p4y)

        p5x = data[x, i, 5, 0]
        p5y = data[x, i, 5, 1]
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

        if has_ball is not tmp_ball:
            if tmp[p] == 0.0:
                if tmp_ball == 0:
                    data[x, tmp_i:i, 0, 0] = data[x, tmp_i:i, 1, 0]
                    data[x, tmp_i:i, 0, 1] = data[x, tmp_i:i, 1, 1]
                elif tmp_ball == 1:
                    data[x, tmp_i:i, 0, 0] = data[x, tmp_i:i, 2, 0]
                    data[x, tmp_i:i, 0, 1] = data[x, tmp_i:i, 2, 1]
                elif tmp_ball == 2:
                    data[x, tmp_i:i, 0, 0] = data[x, tmp_i:i, 3, 0]
                    data[x, tmp_i:i, 0, 1] = data[x, tmp_i:i, 3, 1]
                elif tmp_ball == 3:
                    data[x, tmp_i:i, 0, 0] = data[x, tmp_i:i, 4, 0]
                    data[x, tmp_i:i, 0, 1] = data[x, tmp_i:i, 4, 1]
                elif tmp_ball == 4:
                    data[x, tmp_i:i, 0, 0] = data[x, tmp_i:i, 5, 0]
                    data[x, tmp_i:i, 0, 1] = data[x, tmp_i:i, 5, 1]

                elif tmp_ball == 5:
                    data[x, tmp_i:i, 0, 0] = BASKET_RIGHT[i, 0]
                    data[x, tmp_i:i, 0, 1] = BASKET_RIGHT[i, 1]

                elif tmp_ball == 6:
                    data[x, tmp_i:i, 0, 0] = data[x, tmp_i:i, 0, 0]
                    data[x, tmp_i:i, 0, 1] = data[x, tmp_i:i, 0, 1]

                if i == 0:
                    pass
                else:
                    pass_.append([tmp_i, i])
                    pass_len.append(p_count)
                    p_count = 2
                tmp_i = i
                tmp_ball = has_ball
                count += 1
            else:
                p_count += 1

    if tmp_i < len_[x]-1:
        pass_.append([tmp_i, i])

    get_frame = pass_
    get_frame = np.array(get_frame)

    count = len(get_frame)

    ball = data[x, :, 0, :2]
    p1 = data[x, :, 1, :2]
    p2 = data[x, :, 2, :2]
    p3 = data[x, :, 3, :2]
    p4 = data[x, :, 4, :2]
    p5 = data[x, :, 5, :2]

    new_ball = []
    p11 = []
    p22 = []
    p33 = []
    p44 = []
    p55 = []

    ball_ = []
    new_p1 = []
    new_p2 = []
    new_p3 = []
    new_p4 = []
    new_p5 = []

    dist_frames = []
    total_frames = 0
    l = len_[x]
    if count == 0:
        seg = len_[x]
    if count > 1:
        time = pass_len[0] + 2
        seg = get_frame[0, 1]
        if (seg - time) < 1:
            time = 2
        if seg == 1:
            seg = 2

        new_p1.extend(G.simplify_(p1[get_frame[0, 0]:get_frame[0, 1]], seg))
        new_p2.extend(G.simplify_(p2[get_frame[0, 0]:get_frame[0, 1]], seg))
        new_p3.extend(G.simplify_(p3[get_frame[0, 0]:get_frame[0, 1]], seg))
        new_p4.extend(G.simplify_(p4[get_frame[0, 0]:get_frame[0, 1]], seg))
        new_p5.extend(G.simplify_(p5[get_frame[0, 0]:get_frame[0, 1]], seg))

        #ball seg
        ball_.extend(G.simplify_(ball[get_frame[0, 0]:get_frame[0, 1]], seg))

        ball_new = ball_[:-time]

        ball_old = ball_[-time]
        pass_frame = np.stack((ball_old,ball[get_frame[1, 0]+1]))

        ball_new.extend(G.simplify_(pass_frame, time))
        #ball_new.append(ball_)
        new_ball = np.array(ball_new)
        new_ball = np.reshape(new_ball, newshape=[seg, 2])

        #p1_pos = p1[get_frame[0, 1]:get_frame[1, 0] + 1]
        #new_p1.extend(G.simplify_(p1_pos, time))
        p11.append(new_p1)
        p11 = np.array(p11)
        p11 = np.reshape(p11, newshape=[seg, 2])

        #p2_pos = p2[get_frame[0, 1]:get_frame[1, 0] + 1]
        #new_p2.extend(G.simplify_(p2_pos, time))
        p22.append(new_p2)
        p22 = np.array(p22)
        p22 = np.reshape(p22, newshape=[seg, 2])

        #p3_pos = p3[get_frame[0, 1]:get_frame[1, 0] + 1]
        #new_p3.extend(G.simplify_(p3_pos, time))
        p33.append(new_p3)
        p33 = np.array(p33)
        p33 = np.reshape(p33, newshape=[seg, 2])

        #p4_pos = p4[get_frame[0, 1]:get_frame[1, 0] + 1]
        #new_p4.extend(G.simplify_(p4_pos, time))
        p44.append(new_p4)
        p44 = np.array(p44)
        p44 = np.reshape(p44, newshape=[seg, 2])

        #p5_pos = p5[get_frame[0, 1]:get_frame[1, 0] + 1]
        #new_p5.extend(G.simplify_(p5_pos, time))
        p55.append(new_p5)
        p55 = np.array(p55)
        p55 = np.reshape(p55, newshape=[seg, 2])

        l -= seg

        for k in range(1, count - 1):
            #print("Remaining:", l)
            ball_ = []
            new_p1 = []
            new_p2 = []
            new_p3 = []
            new_p4 = []
            new_p5 = []

            time = pass_len[k] + 2
            #print(get_frame[k,0])
            #print(get_frame[k,1])
            seg = get_frame[k,1] - get_frame[k,0]

            if (seg - time) < 1:
                time = 2
            #print(pass_len[k])
            if seg == 1:
                seg = 2
            l -= seg
            #print(l)
            # print(time)
            #ball_.extend(G.simplify_(ball[get_frame[k, 0]:get_frame[k, 1]], seg - time))
            #ball_.extend(G.simplify_(ball[get_frame[k, 1] - 1:get_frame[k + 1, 0] + 1], time))

            ball_.extend(G.simplify_(ball[get_frame[k, 0]:get_frame[k, 1]], seg))

            ball_new = ball_[:-time]

            ball_old = ball_[-time]

            pass_frame = np.stack((ball_old, ball[get_frame[k+1, 0]]))

            ball_new.extend(G.simplify_(pass_frame, time))

            n_ball = []
            n_ball.append(ball_new)
            n_ball = np.array(n_ball)
            n_ball = np.reshape(n_ball, newshape=[seg, 2])

            new_ball = np.append(new_ball, n_ball, axis=0)

            # new_ball = np.array(new_ball)
            # new_ballx = G.simplify_(ball[get_frame[k,0]:get_frame[k, 1]], seg)
            # new_ball = np.append(new_ball, new_ballx, axis=0)
            new_p1.extend(G.simplify_(p1[get_frame[k, 0]:get_frame[k, 1]], seg))
            #p1_pos = p1[get_frame[k, 1]:get_frame[k + 1, 0] + 1]
            #new_p1.extend(G.simplify_(p1_pos, time))
            n_p1 = []
            n_p1.append(new_p1)
            n_p1 = np.array(n_p1)
            n_p1 = np.reshape(n_p1, newshape=[seg, 2])
            p11 = np.append(p11, n_p1, axis=0)

            # p1g = G.simplify_(p1[get_frame[k,0]:get_frame[k , 1]], seg)
            # p11 = np.append(p11, p1g, axis=0)

            new_p2.extend(G.simplify_(p2[get_frame[k, 0]:get_frame[k, 1]], seg))
            #p2_pos = p2[get_frame[k, 1]:get_frame[k + 1, 0] + 1]
            #new_p2.extend(G.simplify_(p2_pos, time))
            n_p2 = []
            n_p2.append(new_p2)
            n_p2 = np.array(n_p2)
            n_p2 = np.reshape(n_p2, newshape=[seg, 2])
            p22 = np.append(p22, n_p2, axis=0)

            # p2g = G.simplify_(p2[get_frame[k,0]:get_frame[k,1]], seg)
            # p22 = np.append(p22, p2g, axis=0)

            new_p3.extend(G.simplify_(p3[get_frame[k, 0]:get_frame[k, 1]], seg))
            #p3_pos = p3[get_frame[k, 1]:get_frame[k + 1, 0] + 1]
            #new_p3.extend(G.simplify_(p3_pos, time))
            n_p3 = []
            n_p3.append(new_p3)
            n_p3 = np.array(n_p3)
            n_p3 = np.reshape(n_p3, newshape=[seg, 2])
            p33 = np.append(p33, n_p3, axis=0)

            # p3g = G.simplify_(p3[get_frame[k,0]:get_frame[k,1]], seg)
            # p33 = np.append(p33, p3g, axis=0)

            new_p4.extend(G.simplify_(p4[get_frame[k, 0]:get_frame[k, 1]], seg))
            #p4_pos = p4[get_frame[k, 1]:get_frame[k + 1, 0] + 1]
            #new_p4.extend(G.simplify_(p4_pos, time))
            n_p4 = []
            n_p4.append(new_p4)
            n_p4 = np.array(n_p4)
            n_p4 = np.reshape(n_p4, newshape=[seg, 2])
            p44 = np.append(p44, n_p4, axis=0)

            # p4g = G.simplify_(p4[get_frame[k,0]:get_frame[k,1]], seg)
            # p44 = np.append(p44, p4g, axis=0)
            new_p5.extend(G.simplify_(p5[get_frame[k, 0]:get_frame[k, 1]], seg))
            #p5_pos = p5[get_frame[k, 1]:get_frame[k + 1, 0] + 1]
            #new_p5.extend(G.simplify_(p5_pos, time))
            n_p5 = []
            n_p5.append(new_p5)
            n_p5 = np.array(n_p5)
            n_p5 = np.reshape(n_p5, newshape=[seg, 2])
            p55 = np.append(p55, n_p5, axis=0)

            # p5g = G.simplify_(p5[get_frame[k,0]:get_frame[k,1]], seg)
            # p55 = np.append(p55, p5g, axis=0)



        #print("Remaining :", l)
        k = count - 1

        new_ballx = G.simplify_(ball[get_frame[k, 0]:get_frame[k, 1]], l)
        new_ball = np.append(new_ball, new_ballx, axis=0)

        p1g = G.simplify_(p1[get_frame[k, 0]:get_frame[k, 1]], l)
        p11 = np.append(p11, p1g, axis=0)

        p2g = G.simplify_(p2[get_frame[k, 0]:get_frame[k, 1]], l)
        p22 = np.append(p22, p2g, axis=0)

        p3g = G.simplify_(p3[get_frame[k, 0]:get_frame[k, 1]], l)
        p33 = np.append(p33, p3g, axis=0)

        p4g = G.simplify_(p4[get_frame[k, 0]:get_frame[k, 1]], l)
        p44 = np.append(p44, p4g, axis=0)

        p5g = G.simplify_(p5[get_frame[k, 0]:get_frame[k, 1]], l)
        p55 = np.append(p55, p5g, axis=0)
    else:
        seg = len_[x]
        new_ball.append(G.simplify_(ball[get_frame[0, 0]:get_frame[0, 1]], seg))
        new_ball = np.array(new_ball)
        new_ball = np.reshape(new_ball, newshape=[seg, 2])

        p11.append(G.simplify_(p1[get_frame[0, 0]:get_frame[0, 1]], seg))
        p11 = np.array(p11)
        p11 = np.reshape(p11, newshape=[seg, 2])

        p22.append(G.simplify_(p2[get_frame[0, 0]:get_frame[0, 1]], seg))
        p22 = np.array(p22)
        p22 = np.reshape(p22, newshape=[seg, 2])

        p33.append(G.simplify_(p3[get_frame[0, 0]:get_frame[0, 1]], seg))
        p33 = np.array(p33)
        p33 = np.reshape(p33, newshape=[seg, 2])

        p44.append(G.simplify_(p4[get_frame[0, 0]:get_frame[0, 1]], seg))
        p44 = np.array(p44)
        p44 = np.reshape(p44, newshape=[seg, 2])

        p55.append(G.simplify_(p5[get_frame[0, 0]:get_frame[0, 1]], seg))
        p55 = np.array(p55)
        p55 = np.reshape(p55, newshape=[seg, 2])


    ndata = new_ball
    ndata = np.array(ndata)
    ndata = np.append(ndata, p11, axis=1)
    ndata = np.append(ndata, p22, axis=1)
    ndata = np.append(ndata, p33, axis=1)
    ndata = np.append(ndata, p44, axis=1)
    ndata = np.append(ndata, p55, axis=1)

    #if variable length
    '''
    if 235-len_[x] > 0:
        diff_ = 235-len_[x]
        zero_ = np.zeros([235,12])
        ndata = np.vstack((ndata,zero_[:diff_]))

        save_ball.append(ndata)
    else:
        print("Long seq:",len_[x])
        save_ball.append(ndata)
    '''

    # data[x] = ndata
    save_ball.append(ndata)

save_ball = np.array(save_ball)
print(save_ball.shape)

# print(data.shape)
np.save(output_file, save_ball)