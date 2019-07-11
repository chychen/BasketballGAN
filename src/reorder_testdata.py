from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import h5py
import numpy as np
import shutil
import os


def length_fn(vec, axis):
    return np.sqrt(np.sum(vec[:, :, :, 0:2]**2, axis=axis))


def main():
    test_path = '/workspace/data/nctu_cgvlab_bballgan/Reordered_Data/Test/'
    real = np.load(test_path+'TestReal.npy')
    real_cond = np.load(test_path+'TestRealCond.npy')
    seq = np.reshape(np.load(test_path+'TestSeq.npy'), [real.shape[0],real.shape[1],6,2])
    seq_cond = np.load(test_path+'TestSeqCond.npy')
    length = np.load(test_path+'TestLength.npy')
    print(real.shape)
    print(real_cond.shape)
    print(seq.shape)
    print(seq_cond.shape)
    print(length.shape)

    # 1. find the priority of offensive players by summing the distance to ball, then we get an order (off_order).
    off_ball_obs = real[:, :, 0:1]
    off_pl_obs_first = real[:, :, 1:6]
    len_ball_off = np.sum(
        length_fn(off_ball_obs - off_pl_obs_first, axis=-1), axis=1)
    off_order = np.argsort(len_ball_off, axis=-1)
    print(off_order.shape)
    # 2. accumulate distance of the whole episode between 5 offensive players to each defensive player, then we get a 5(def)*5(off) matrix (acc_def_mat).
    off_pl_obs = real[:, :, 1:6]
    def_pl_obs = real[:, :, 6:11]
    acc_def_mat = []
    for def_pl_id in range(5):
        one_def_pl_obs = def_pl_obs[:, :, def_pl_id:def_pl_id + 1]
        len_onedef_off = length_fn(off_pl_obs - one_def_pl_obs, axis=-1)
        tmp_sum = np.sum(len_onedef_off, axis=1)
        acc_def_mat.append(tmp_sum)
    acc_def_mat = np.stack(acc_def_mat, axis=1)
    print(acc_def_mat.shape)
    # 3. according to (off_order), we paired offense with the best defender (paired_def_order) from (acc_def_mat) sequencially.
    paired_def_order = []
    for i, off_order_one_epi in enumerate(off_order):
        tmp = []
        for off_id in off_order_one_epi:
            acc_def_one_off = acc_def_mat[i, :, off_id]
            best_def_order = np.argsort(acc_def_one_off, axis=-1)
            for best_def in best_def_order:
                if best_def in tmp:
                    continue
                else:
                    tmp.append(best_def)
                    break
        paired_def_order.append(tmp)
    paired_def_order = np.array(paired_def_order)
    print(paired_def_order.shape)
    # 4. make sure data_act, data_vel_act and data_init_vel are consistent with real
    data_defense_obs = real[:, :, 6:11]
    ori_data_defense_obs = np.array(real[:, :, 6:11])
    for i in range(data_defense_obs.shape[0]):
        for def_id in range(5):
            data_defense_obs[i, :, off_order[
                i, def_id]] = ori_data_defense_obs[
                    i, :, paired_def_order[i, def_id]]
    assert (real[:, :, 6:11] != ori_data_defense_obs
            ).any(), "real(defense) should be modified"
    # 5. finally, we use (off_order) to permute the index in data.
    data_offense_obs = real[:, :, 1:6]
    data_defense_obs = real[:, :, 6:11]
    data_real_cond = real_cond[:, :, 1:6]
    data_seq = seq[:, :, 1:6]
    data_seq_cond = seq_cond[:, :, 1:6]
    ori_data_offense_obs = np.array(real[:, :, 1:6])
    ori_data_defense_obs = np.array(real[:, :, 6:11])
    ori_data_real_cond = np.array(real_cond[:, :, 1:6])
    ori_data_seq = np.array(seq[:, :, 1:6])
    ori_data_seq_cond = np.array(seq_cond[:, :, 1:6])
    for i in range(data_defense_obs.shape[0]):
        for off_id in range(5):
            data_offense_obs[i, :, off_id] = ori_data_offense_obs[
                i, :, off_order[i, off_id]]
            data_defense_obs[i, :, off_id] = ori_data_defense_obs[
                i, :, off_order[i, off_id]]
            data_real_cond[i, :, off_id] = ori_data_real_cond[
                i, :, off_order[i, off_id]]
            data_seq[i, :, off_id] = ori_data_seq[
                i, :, off_order[i, off_id]]
            data_seq_cond[i, :, off_id] = ori_data_seq_cond[
                i, :, off_order[i, off_id]]
    assert (real[:, :, 1:6] != ori_data_offense_obs
            ).any(), "real(offense) should be modified"
    assert (real[:, :, 6:11] != ori_data_defense_obs
            ).any(), "real(defense) should be modified"

    # 6. save files into h5py
    save_path = '/workspace/data/nctu_cgvlab_bballgan/Reordered_Data/Test/Reordered/'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print('rm -rf "%s" complete!' % save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path+'TestReal.npy', real)
    np.save(save_path+'TestRealCond.npy', real_cond)
    np.save(save_path+'TestSeq.npy', np.reshape(seq, [seq.shape[0],seq.shape[1],12]))
    np.save(save_path+'TestSeqCond.npy', seq_cond)
    np.save(save_path+'TestLength.npy', length)
        
if __name__ == "__main__":
    main()
