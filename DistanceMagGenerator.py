# -*- coding:utf-8 -*-
# Created by steve @ 17-11-21 下午9:36
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

from MagPreprocess import MagPreprocess

import seaborn as sns


import timeit
import time

if __name__ == '__main__':
    sns.set('paper','white')

    start_time = time.time()
    dir_name = '/home/steve/Data/II/17/'

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

    ax.plot(v_data[:, 12], v_data[:, 13], v_data[:, 14], '-*',label='trace 3d \\alpha ')
    ax.legend()

    mDetector = MagPreprocess.MagDetector(v_data[:, 8:11],
                                          v_data[:, 2:5],
                                          v_data[:, 12:],
                                          v_data[:, 11])

    mDetector.Step2Length(False)
    # mDetector.GetFFTDis(20.0)
    # mDetector.MultiLayerNormFFt([30.0, 25.0, 20.0, 15.0, 10.0, 5.0])
    # mDetector.GetDirectDis(20.0)
    mDetector.GetZValue(False)
    mDetector.ConvertMagAttitude()
    # mDetector.GetZFFtDis(20.0)
    # mDetector.MultiLayerNZFFt([30, 25, 20.0, 15.0, 10.0, 5.0])
    # mDetector.GetRelativeAttDis(50.0)
    mDetector.MultiLayerANZFFt([30.0, 25.0, 20.0, 15.0, 10.0, 5.0])

    the_threshold = 30
    max_dis = 30.0

    for i in range(0, mDetector.tmp_fft_mat.shape[0]):
        for j in range(i + 1, mDetector.tmp_fft_mat.shape[0]):
            tttttttta = 1
    #         # if mDetector.tmp_fft_mat[i, j] < the_threshold and \
    #         #                 abs(mDetector.length_array[i] - mDetector.length_array[j]) > 50.0 and \
    #         #                 mDetector.tmp_fft_mat[i, j] > 2.0 and \
    #         #                 abs(v_data[i, 11] - v_data[j, 11]) < 1e10:
    #         #
    #         # if mDetector.tmp_src_mat[i, j] < 450.0 and \
    #         #                 np.linalg.norm(mDetector.tmp_src_mat[i, j] - mDetector.tmp_src_mat[i,
    #         #                                                              j - 10:j + 10]) > 1000 and \
    #         #                 abs(mDetector.length_array[i] - mDetector.length_array[j]) > 50.0 and \
    #         #                 mDetector.tmp_fft_mat[i, j] > 2.0 and \
    #         #                 abs(v_data[i, 11] - v_data[j, 11]) < 1e10:
    #         # if mDetector.tmp_mul_mat[i, j] < the_threshold and \
    #         #                 np.mean(np.abs(mDetector.tmp_mul_mat[i, j - 20:j + 20] - mDetector.tmp_mul_mat[
    #         #                     i, j])) > 5000 and \
    #         #                 abs(mDetector.length_array[i] - mDetector.length_array[j]) > 30.0 and \
    #         #                 abs(v_data[i, 11] - v_data[j, 11]) < 1e10 and \
    #         #                 mDetector.length_array[i] > max_dis and mDetector.length_array[j] > max_dis and \
    #         #                         mDetector.length_array[-1] - mDetector.length_array[i] > max_dis and \
    #         #                         mDetector.length_array[-1] - mDetector.length_array[j] > max_dis:
    #
    #         # if mDetector.tmp_mnz_mat[i, j] < the_threshold and \
    #         #                 abs(mDetector.length_array[i] - mDetector.length_array[j]) > max_dis and \
    #         #                 abs(v_data[i, 11] - v_data[j, 11]) < 1e11:

            if mDetector.tmp_mnza_mat[i, j] > 5.0 and \
                    abs(mDetector.length_array[i] - mDetector.length_array[j]) > max_dis and \
                    abs(v_data[i, 11] - v_data[j, 11]) < 1e11:

                ax.plot(
                    [v_data[i, 12], v_data[j, 12]],
                    [v_data[i, 13], v_data[j, 13]],
                    [v_data[i, 14], v_data[j, 14]],
                    'r--',
                    linewidth=0.1  # p.log2(mDetector.tmp_fft_mat[i,j])[0,0]
                )
    # plt.figure()
    # plt.title('mag att feature')
    # plt.imshow(mDetector.mag_att_feature.transpose())
    # plt.colorbar()
    print('totally time :', time.time() - start_time)

    # plt.figure()
    # plt.title('hist of dis')
    # plt.hist(mDetector.tmp_mnza_mat.reshape([-1]), bins=60)

    plt.show()
