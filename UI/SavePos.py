import numpy as np
import Bezier
from math import sqrt

def save_pos(data):
    print("Saving")
    seg = np.array(data)
    seg_l = len(seg)
    print("Segments:",seg_l)

    a_pass = np.array(seg[0, 0])
    b_l = len(seg[0][0])
    b_pos = seg[0][0][:b_l - 3]
    pass_pos = seg[0][0][b_l - 2:]

    time = get_ball_speed(pass_pos)
    l = get_seg_len(seg[0])

    if l < 3:
        l = 4
    print("Seq frame:", l)
    # Smooth trajectory, apply bezier curve
    # check if is a pass
    if check_pass(a_pass):
        ballx1, bally1 = Bezier.plotB(b_pos, nTimes=l)

        pass_ = [[ballx1[time-1],bally1[time-1]],pass_pos[1]]
        ballx, bally = Bezier.plotB(pass_, nTimes=time)

        ballx = np.append(ballx, ballx1[time:])
        bally = np.append(bally, bally1[time:])

        p = np.column_stack([ballx, bally])
        p = p[::-1]
        ball = np.array(p)
    # not a pass
    else:
        ballx, bally = Bezier.plotB(seg[0, 0], nTimes=l)
        p = np.column_stack([ballx, bally])
        p = p[::-1]
        ball = p

    # Smooth trajectory if pass avaliable
    x1, y1 = Bezier.plotB(seg[0, 1], nTimes=l)
    x2, y2 = Bezier.plotB(seg[0, 2], nTimes=l)
    x3, y3 = Bezier.plotB(seg[0, 3], nTimes=l)
    x4, y4 = Bezier.plotB(seg[0, 4], nTimes=l)
    x5, y5 = Bezier.plotB(seg[0, 5], nTimes=l)
    #Stack and save trajectory

    p = np.column_stack([x1, y1])
    p = p[::-1]
    p1 = p

    p = np.column_stack([x2, y2])
    p = p[::-1]
    p2 = p

    p = np.column_stack([x3, y3])
    p = p[::-1]
    p3 = p

    p = np.column_stack([x4, y4])
    p = p[::-1]
    p4 = p

    p = np.column_stack([x5, y5])
    p = p[::-1]
    p5 = p

    #segment smoothing
    for i in range(1, seg_l):
        a_pass = np.array(seg[i, 0])
        l = get_seg_len(seg[i])

        if l < 3:
            l = 4

        if check_pass(a_pass):
            b_l = len(seg[i][0])
            b_pos = seg[i][0][:b_l - 3]
            pass_pos = seg[i][0][b_l - 2:]
            #pass speed
            time = get_ball_speed(pass_pos)
            if time < 3:
                time = 4

            ballx1, bally1 = Bezier.plotB(b_pos, nTimes=l)
            pass_ = [[ballx1[time - 1], bally1[time - 1]], pass_pos[1]]

            ballx, bally = Bezier.plotB(pass_, nTimes=time)

            ballx = np.append(ballx, ballx1[time:])
            bally = np.append(bally, bally1[time:])

            p = np.column_stack([ballx, bally])
            p = p[::-1]
            ball2 = np.array(p)

            ball = np.append(ball, ball2, axis=0)
        else:
            ballx, bally = Bezier.plotB(seg[i, 0], nTimes=l)
            p = np.column_stack([ballx, bally])
            p = p[::-1]
            ball = np.append(ball, p, axis=0)

        x1, y1 = Bezier.plotB(seg[i, 1], nTimes=l)
        x2, y2 = Bezier.plotB(seg[i, 2], nTimes=l)
        x3, y3 = Bezier.plotB(seg[i, 3], nTimes=l)
        x4, y4 = Bezier.plotB(seg[i, 4], nTimes=l)
        x5, y5 = Bezier.plotB(seg[i, 5], nTimes=l)

        p = np.column_stack([x1, y1])
        p = p[::-1]
        p1 = np.append(p1, p, axis=0)

        p = np.column_stack([x2, y2])
        p = p[::-1]
        p2 = np.append(p2, p, axis=0)

        p = np.column_stack([x3, y3])
        p = p[::-1]
        p3 = np.append(p3, p, axis=0)

        p = np.column_stack([x4, y4])
        p = p[::-1]
        p4 = np.append(p4, p, axis=0)

        p = np.column_stack([x5, y5])
        p = p[::-1]
        p5 = np.append(p5, p, axis=0)
        ####################################

    print("Stacking")

    pos = np.append(ball, p1, axis=1)
    pos = np.append(pos,p2,axis = 1)
    pos = np.append(pos, p3, axis=1)
    pos = np.append(pos, p4, axis=1)
    pos = np.append(pos, p5, axis=1)

    '''
    if p1_full.size == 0:
        pos = np.append(ball, p1, axis=1)
    else:
        pos = np.append(ball,p1_full,axis=1)

    if p2_full.size == 0:
        pos = np.append(pos, p2, axis=1)
    else:
        pos = np.append(pos,p2_full,axis = 1)

    if p3_full.size == 0:
        pos = np.append(pos, p3, axis=1)
    else:
        pos = np.append(pos, p3_full, axis=1)

    if p4_full.size == 0:
        pos = np.append(pos, p4, axis=1)
    else:
        pos = np.append(pos, p4_full, axis=1)

    if p5_full.size == 0:
        pos = np.append(pos, p5, axis=1)
    else:
        pos = np.append(pos, p5_full, axis=1)
    '''
    pos = np.divide(pos, 10)

    np.save('./Points/points2.npy', pos)
    print("Saved pos")
    print(pos.shape)

def Distance(p1, p2):
    d = sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    return d

def check_pass(pos):
    l = len(pos)
    for i in range(l - 1):
        if pos[i, 0] == -1:
            return True

    return False

def get_ball_speed(pass_pos):
    dist_ = Distance(pass_pos[0], pass_pos[1])
    speed_ = dist_ / 240.0
    time = int(speed_ / 0.2)
    print("Ball distance:", dist_)
    print("Frames:", time)

    return time

def get_seg_len(seg):
    player_dist = []

    for k in range(1, 6):
        player = seg[k]
        pd = 0.
        for j in range(len(player) - 1):
            pd += Distance(player[j], player[j + 1])
        player_dist.append(pd)

    player_dist.sort()
    print(player_dist)
    player_speed = player_dist[-1] / 70.0
    len_ = int(player_speed / 0.2)

    return len_

def get_traj(seg, l):
    pos = []
    for i in seg:
        pos.extend(i)

    x, y = Bezier.plotB(pos, nTimes=l)
    p = np.column_stack([x, y])
    p = p[::-1]

    return p