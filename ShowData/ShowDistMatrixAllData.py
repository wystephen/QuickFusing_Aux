# -*- coding:utf-8 -*-
# Created by steve @ 17-12-1 下午1:33
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
import time

if __name__ == '__main__':

    dir_list = [16,17,20,28,34,35]
    dir_list = [16,17]
    plt.figure()
    plot_rows = 3
    plot_cols = len(dir_list)
    # plt.subplot()
    plt.figure()
    for dir_i in range(len(dir_list)):
        dir_str = dir_list[dir_i]
        dir_name = '/home/steve/Data/II/'+str(dir_str)+"/"

        v_data = np.loadtxt(dir_name + 'vertex_all_data.csv', delimiter=',')

        '''
                id | time ax ay az wx wy wz mx my mz pressure| x  y  z  vx vy vz| qx qy qz qw
                0  |   1   2  3 4  5   6  7 8  9  10 11      | 12 13 14 15 16 17| 18 19 20 21
                1 + 11 + 6 + 4 = 22
            '''
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

        plt.subplot(plot_rows,plot_cols,dir_i+1)
        plt.title(str(dir_str)+' trace')
        plt.plot(v_data[:,12],v_data[:,13],'r--+')
        plt.grid()
        plt.subplot(plot_rows,plot_cols,dir_i+plot_cols+1)
        plt.title(str(dir_str)+' dis_matrix')
        plt.imshow(mDetector.tmp_mnza_mat)
        plt.colorbar()



    plt.show()