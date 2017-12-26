# -*- coding:utf-8 -*-
# Created by steve @ 17-12-26 上午10:11
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
    # sns.set('paper','white')

    start_time = time.time()
    dir_name = '/home/steve/Data/II/35/'

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
    mDetector.GetZValue(ifshow=False)
    mDetector.MultiLayerANZFFt([10,5,2],ifshow=True)
    # mDetector.GetDirectDis(True)


    plt.figure()
    x = np.linspace(mDetector.length_array[5],
                    mDetector.length_array[-5],
                    mDetector.length_array.shape[0])
    plt.plot(x,mDetector.f(x),'*-',label='norm')
    plt.plot(x,mDetector.zf(x),'*-', label='z-norm')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(x,mDetector.f(x),'*-',label='before')
    plt.plot(x,mDetector.f(x+0.5),'*-',label='after')
    plt.plot(x,np.abs(mDetector.f(x+0.5)-mDetector.f(x)),'*--',label='diff')
    plt.legend()


    before = mDetector.f(x)
    after = mDetector.f(x+0.5)

    length_div_2 = 15



    plt.grid()
    plt.show()

