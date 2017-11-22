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


if __name__ == '__main__':
    v_data = np.loadtxt('/home/steve/Data/II/30/vertex_all_data.csv', delimiter=',')

    '''
            id | time ax ay az wx wy wz mx my mz pressure| x  y  z  vx vy vz| qx qy qz qw
            0  |   1   2  3 4  5   6  7 8  9  10 11      | 12 13 14 15 16 17| 18 19 20 21
            1 + 11 + 6 + 4 = 22
        '''

    plt.figure()
    plt.grid()
    plt.title('trace')
    plt.plot(v_data[:,12],v_data[:,13],'r-*')

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    ax.plot(v_data[:,12],v_data[:,13],v_data[:,14],'r-*')


    mDetector = MagPreprocess.MagDetector(v_data[:,8:11],
                            v_data[:,12:15],
                            v_data[:,11])

    mDetector.Step2Length(10.0)





    plt.show()