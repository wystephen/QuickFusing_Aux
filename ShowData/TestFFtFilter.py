# -*- coding:utf-8 -*-
# Created by steve @ 17-12-8 上午10:32
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

import cv2

if __name__ == '__main__':
    # sns.set('paper', 'white')

    start_time = time.time()
    dir_name = '/home/steve/Data/II/34/'

    load_start = time.time()
    v_data = np.loadtxt(dir_name + 'vertex_all_data.csv', delimiter=',')
    load_end = time.time()

    print('load file costs time :', load_end-load_start)

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
    # mDetector.GetDirectDis(20.0)
    mDetector.GetZValue(False)
    mDetector.ConvertMagAttitude()
    # mDetector.GetZFFtDis(20.0)
    # mDetector.MultiLayerNZFFt([30, 25, 20.0, 15.0, 10.0, 5.0])
    # mDetector.GetRelativeAttDis(50.0)
    mDetector.MultiLayerANZFFt([30, 20.0, 10.0, 5.0])


    t = np.copy(mDetector.tmp_mnza_mat)


    rate = 1.0/8.0
    mask_2d = np.array(
        [
            [0.0, 0.0, rate, 0.0, 0.0],
            [0.0, 0.0, rate, 0.0, 0.0],
            [rate, rate, 1.0, rate, rate],
            [0.0, 0.0, rate, 0.0, 0.0],
            [0.0, 0.0, rate, 0.0, 0.0]
        ]
    )

    t = cv2.filter2D(t,-1,mask_2d)





    plt.figure()
    plt.imshow(t)
    plt.colorbar()
    plt.show()