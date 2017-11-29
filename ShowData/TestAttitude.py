# -*- coding:utf-8 -*-
# Created by steve @ 17-11-29 下午3:21
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

from transforms3d.euler import quat2euler

if __name__ == '__main__':
    dir_name = '/home/steve/Data/II/30/'

    ### key 16 17 20 ||| 28  30  (31)
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

    mDetector.Step2Length()
    # mDetector.GetFFTDis(20.0)
    # mDetector.MultiLayerNormFFt([30.0, 25.0, 20.0, 15.0, 10.0, 5.0])
    # mDetector.GetDirectDis(20.0)
    mDetector.GetZValue(False)
    mDetector.ConvertMagAttitude()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.plot(v_data[:, 12], v_data[:, 13], v_data[:, 14], 'r-*')

    qeuler = np.zeros([v_data.shape[0],3])
    for i in range(v_data.shape[0]):
        qeuler[i,:] = quat2euler([v_data[i,21],v_data[i,18],v_data[i,19],v_data[i,20]])

    pangle = np.zeros([v_data.shape[0],3])
    for i in range(v_data.shape[0]-1):
        t = v_data[i+1,12:15]-v_data[i,12:15]
        pangle[i,2] = np.arctan2(t[1],t[0])


    mangle = mDetector.angle
    relative_mangle = mangle[:,0]-pangle[:,2]+qeuler[:,2]



    qeuler /= (np.pi / 180.0)
    pangle /= (np.pi /180.0)


    plot_row = 2
    plot_col = 3
    plt.figure()
    plt.subplot(plot_row,plot_col,1)
    plt.title('qeuler')
    for i in range(qeuler.shape[1]):
        plt.plot(qeuler[:,i],'-+',label=str(i))
    plt.legend();plt.grid()


    # plt.figure()
    plt.subplot(plot_row,plot_col,2)
    plt.title('pangle')
    for i in range(pangle.shape[1]):
        plt.plot(pangle[:,i],'-+',label=str(i))
    plt.legend();plt.grid()


    # plt.figure()
    plt.subplot(plot_row,plot_col,3)
    plt.title('diff')
    for i in range(pangle.shape[1]):
        plt.plot(qeuler[:,i]-pangle[:,i],'-+',label=str(i))
    plt.legend();plt.grid()



    # plt.figure()
    plt.subplot(plot_row,plot_col,4)
    plt.title('trace 2d')
    plt.plot(v_data[:,12],v_data[:,13],'r--+')
    plt.grid()

    plt.subplot(plot_row,plot_col,5)
    plt.title('angle ')
    plt.plot(mangle[:,0],label='mag angle')
    plt.plot(relative_mangle,label='relative mag angle')
    plt.grid()
    plt.legend()



    plt.show()
