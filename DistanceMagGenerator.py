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

import timeit

if __name__ == '__main__':
    dir_name = '/home/steve/Data/II/32/'
    v_data = np.loadtxt(dir_name + 'vertex_all_data.csv', delimiter=',')

    '''
            id | time ax ay az wx wy wz mx my mz pressure| x  y  z  vx vy vz| qx qy qz qw
            0  |   1   2  3 4  5   6  7 8  9  10 11      | 12 13 14 15 16 17| 18 19 20 21
            1 + 11 + 6 + 4 = 22
        '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(v_data[:, 12], v_data[:, 13], v_data[:, 14], 'r-*')

    mDetector = MagPreprocess.MagDetector(v_data[:, 8:11],
                                          v_data[:, 12:15],
                                          v_data[:, 11])

    mDetector.Step2Length()
    mDetector.GetFFTDis(20.0)
    mDetector.MultiLayerFFt([30.0, 20.0, 10.0, 5.0])
    mDetector.GetDirectDis(20.0)

    the_threshold = 3500

    for i in range(0, mDetector.tmp_fft_mat.shape[0]):
        for j in range(i, mDetector.tmp_fft_mat.shape[0]):
            if mDetector.tmp_fft_mat[i, j] < the_threshold and \
                            abs(mDetector.length_array[i] - mDetector.length_array[j]) > 50.0 and \
                            mDetector.tmp_fft_mat[i, j] > 2.0 and \
                            abs(v_data[i, 11] - v_data[j, 11]) < 1e10:
            #
            # if mDetector.tmp_src_mat[i, j] < 450.0 and \
            #                 np.linalg.norm(mDetector.tmp_src_mat[i, j] - mDetector.tmp_src_mat[i,
            #                                                              j - 10:j + 10]) > 1000 and \
            #                 abs(mDetector.length_array[i] - mDetector.length_array[j]) > 50.0 and \
            #                 mDetector.tmp_fft_mat[i, j] > 2.0 and \
            #                 abs(v_data[i, 11] - v_data[j, 11]) < 1e10:
                ax.plot(
                    [v_data[i, 12], v_data[j, 12]],
                    [v_data[i, 13], v_data[j, 13]],
                    [v_data[i, 14], v_data[j, 14]],
                    'b--',
                    linewidth=0.1  # p.log2(mDetector.tmp_fft_mat[i,j])[0,0]
                )

    plt.show()
