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

import math

if __name__ == '__main__':
    sns.set('paper', 'white')

    tmp_mnza_mat = np.loadtxt('19mnza_mat.data')
    tmp_src_mat = np.loadtxt('19source_distance_mat.data')
    trace_graph = np.loadtxt('test.txt', delimiter=',')

    ref_dis_mat = np.zeros([trace_graph.shape[0], trace_graph.shape[0]])

    ref_dis_mat = squareform(pdist(trace_graph))
    ref_dis_mat = np.where(ref_dis_mat < 1.6, 1.0, 0.0)

    threshold_list = np.asarray(range(0, 6000, 1), dtype=np.float)
    threshold_list = threshold_list / 10.0
    # print(threshold_list)

    fft_TPR = np.zeros(threshold_list.shape[0])
    fft_FPR = np.zeros(threshold_list.shape[0])

    src_TPR = np.zeros(threshold_list.shape[0])
    src_FPR = np.zeros(threshold_list.shape[0])

    real_positive = float(ref_dis_mat[ref_dis_mat > 0.5].shape[0])
    real_negative = float(ref_dis_mat[ref_dis_mat < 0.5].shape[0])

    print('real positive:', real_positive, ' real negative: ', real_negative)
    tmp_feature_mat = tmp_mnza_mat

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

    tmp_feature_mat = tmp_src_mat

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
    plt.plot(fft_FPR, fft_TPR, '*', label='fft')
    plt.plot(src_FPR, fft_TPR, '*', label='src')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('TPR')
    plt.plot(threshold_list, fft_TPR, label='fft')
    plt.plot(threshold_list, src_TPR, label='src')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('FPR')
    plt.plot(threshold_list, fft_FPR, label='fft')
    plt.plot(threshold_list, src_FPR, label='srt')
    plt.legend()

    plt.figure()
    plt.plot(threshold_list, fft_TPR / fft_FPR, label='fft')
    plt.plot(threshold_list, src_TPR / fft_FPR, label='src')
    plt.legend()
    plt.grid()

    t_list = np.linspace(0.0, 1.8, 30)
    plt.figure()
    for index, threshold in enumerate(t_list):
        print(index, threshold)
        plt.subplot(3, math.ceil(float(t_list.shape[0]) / 3), index + 1)
        tmp_mat = np.vectorize(lambda x: 0.0 if x < threshold else 1.0)(tmp_src_mat)
        plt.imshow(tmp_mat)
        plt.title('threshold:' + str(threshold))

    t_list = np.linspace(0.0, 5.0, 30)
    plt.figure()
    for index, threshold in enumerate(t_list):
        print(index, threshold)
        plt.subplot(3, math.ceil(float(t_list.shape[0]) / 3), index + 1)
        tmp_mat = np.vectorize(lambda x: 0.0 if x < threshold else 1.0)(tmp_mnza_mat)
        plt.imshow(tmp_mat)
        plt.title('threshold:' + str(threshold))

    plt.show()
