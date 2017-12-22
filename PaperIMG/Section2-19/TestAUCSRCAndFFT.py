# -*- coding:utf-8 -*-
# Created by steve @ 17-12-22 下午9:22
'''
                   _ooOoo_ 
                  o8888888o 
                  88" . "88 
                  (| -_- |) 
                  O\  =  /O 
               ____/`---'\____ 
             .'  \\|     |//  `. 
            /  \\|||  :  |||//  \ 
           /  _||||| -:- |||||-  \ 
           |   | \\\  -  /// |   | 
           | \_|  ''\---/''  |   | 
           \  .-\__  `-`  ___/-. / 
         ___`. .'  /--.--\  `. . __ 
      ."" '<  `.___\_<|>_/___.'  >'"". 
     | | :  `- \`.;`\ _ /`;.`/ - ` : | | 
     \  \ `-.   \_ __\ /__ _/   .-` /  / 
======`-.____`-.___\_____/___.-`____.-'====== 
                   `=---=' 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
         佛祖保佑       永无BUG 
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.distance import pdist, squareform

from MagPreprocess import MagPreprocess

import seaborn as sns

import timeit
import time

if __name__ == '__main__':
    sns.set('paper', 'white')

    start_time = time.time()
    dir_name = '/home/steve/Data/II/19/'

    ### key 16 17 20 ||| 28  30  (31)
    ##  33 34 35

    v_data = np.loadtxt(dir_name + 'vertex_all_data.csv', delimiter=',')

    '''
            id | time ax ay az wx wy wz mx my mz pressure| x  y  z  vx vy vz| qx qy qz qw
            0  |   1   2  3 4  5   6  7 8  9  10 11      | 12 13 14 15 16 17| 18 19 20 21
            1 + 11 + 6 + 4 = 22
        '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(v_data[:, 12], v_data[:, 13], v_data[:, 14], '-*', label='trace 3d \\alpha ')
    ax.legend()

    mDetector = MagPreprocess.MagDetector(v_data[:, 8:11],
                                          v_data[:, 2:5],
                                          v_data[:, 12:],
                                          v_data[:, 11])

    mDetector.Step2Length(False)
    # mDetector.GetFFTDis(20.0)
    # mDetector.MultiLayerNormFFt([30.0, 25.0, 20.0, 15.0, 10.0, 5.0])
    mDetector.GetDirectDis(30.0)
    mDetector.GetZValue(False)
    mDetector.ConvertMagAttitude()
    # mDetector.GetZFFtDis(20.0)
    # mDetector.MultiLayerNZFFt([30, 25, 20.0, 15.0, 10.0, 5.0])
    # mDetector.GetRelativeAttDis(50.0)
    mDetector.MultiLayerANZFFt([30.0, 25, 20, 15, 10, 5.0])

    trace_graph = np.loadtxt('test.txt', delimiter=',')

    ref_dis_mat = np.zeros([trace_graph.shape[0], trace_graph.shape[0]])

    ref_dis_mat = squareform(pdist(trace_graph))
    ref_dis_mat = np.where(ref_dis_mat < 1.6, 1.0, 0.0)

    threshold_list = np.asarray(range(0, 600, 1), dtype=np.float)
    threshold_list = threshold_list / 10.0
    # print(threshold_list)

    fft_TPR = np.zeros(threshold_list.shape[0])
    fft_FPR = np.zeros(threshold_list.shape[0])

    src_TPR = np.zeros(threshold_list.shape[0])
    src_FPR = np.zeros(threshold_list.shape[0])

    real_positive = float(ref_dis_mat[ref_dis_mat > 0.5].shape[0])
    real_negative = float(ref_dis_mat[ref_dis_mat < 0.5].shape[0])

    print('real positive:', real_positive, ' real negative: ', real_negative)
    tmp_feature_mat = mDetector.tmp_mnza_mat

    mnza_mat_without_line = np.zeros_like(tmp_feature_mat)
    for i in range(tmp_feature_mat.shape[0]):
        for j in range(tmp_feature_mat.shape[1]):
            if abs(i - j) < 5:
                mnza_mat_without_line[i, j] = tmp_feature_mat.max()
                if ref_dis_mat[i, j] > 0.5:
                    real_positive -= 1.0
                else:
                    real_negative -= 1.0
            else:
                mnza_mat_without_line[i, j] = tmp_feature_mat[i, j]

    for i in range(2, threshold_list.shape[0]):
        x_list, y_list = np.where(mnza_mat_without_line < threshold_list[i])
        tmp = ref_dis_mat[x_list, y_list]

        tp = tmp[tmp > 0.5].shape[0]
        fp = tmp[tmp < 0.5].shape[0]

        fft_TPR[i] = float(tp) / real_positive
        fft_FPR[i] = float(fp) / real_negative

    tmp_feature_mat = mDetector.tmp_src_mat

    mnza_mat_without_line = np.zeros_like(tmp_feature_mat)
    for i in range(tmp_feature_mat.shape[0]):
        for j in range(tmp_feature_mat.shape[1]):
            if abs(i - j) < 5:
                mnza_mat_without_line[i, j] = tmp_feature_mat.max()
                if ref_dis_mat[i, j] > 0.5:
                    real_positive -= 1.0
                else:
                    real_negative -= 1.0
            else:
                mnza_mat_without_line[i, j] = tmp_feature_mat[i, j]

    for i in range(2, threshold_list.shape[0]):
        x_list, y_list = np.where(mnza_mat_without_line < threshold_list[i])
        tmp = ref_dis_mat[x_list, y_list]

        tp = tmp[tmp > 0.5].shape[0]
        fp = tmp[tmp < 0.5].shape[0]

        src_TPR[i] = float(tp) / real_positive
        src_FPR[i] = float(fp) / real_negative

    plt.figure()
    plt.plot(fft_FPR,fft_TPR,'*',label='fft')
    plt.plot(src_FPR,fft_TPR,'*',label='src')
    plt.legend()
    plt.grid()
    plt.show()
