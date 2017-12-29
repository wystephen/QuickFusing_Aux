# -*- coding:utf-8 -*-
# Created by steve @ 17-12-29 下午6:17
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
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    pair_mat = np.loadtxt('./DCS/corner_pair.txt', delimiter=',')
    pair_mat = pair_mat.astype(dtype=np.int)

    dcs_trace = np.loadtxt('./DCS/test.txt', delimiter=',')
    index_list = np.linspace(0, dcs_trace.shape[0], dcs_trace.shape[0])

    robust_trace = np.loadtxt('./Robust/test.txt', delimiter=',')

    fig = plt.figure()

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(dcs_trace[:, 0], dcs_trace[:, 1], index_list, '-+')
    for index, v in enumerate(pair_mat):
        ax1.plot(np.asarray([dcs_trace[v[0], 0], dcs_trace[v[1], 0]]),
                 np.asarray([dcs_trace[v[0], 1], dcs_trace[v[1], 1]]),
                 np.asarray([index_list[v[0]], index_list[v[1]]]),
                 '-g')

    ax = fig.add_subplot(132, projection='3d')
    ax.plot(robust_trace[:, 0], robust_trace[:, 1], index_list, '-+')
    for index, v in enumerate(pair_mat):
        ax.plot(np.asarray([robust_trace[v[0], 0], robust_trace[v[1], 0]]),
                np.asarray([robust_trace[v[0], 1], robust_trace[v[1], 1]]),
                np.asarray([index_list[v[0]], index_list[v[1]]]),
                '-g')

    ax = fig.add_subplot(133, projection='3d')
    ax.plot(robust_trace[516:550, 0], robust_trace[516:550, 1], index_list[516:550], 'b-+')
    ax.plot(robust_trace[225:250, 0], robust_trace[225:250, 1], index_list[225:250], 'b-+')
    # ax.plot(dcs_trace[200:550, 0], dcs_trace[200:550, 1], index_list[200:550], '-+')
    for index, v in enumerate(pair_mat):
        if 200 < index < 550:
            ax.plot(np.asarray([robust_trace[v[0], 0], robust_trace[v[1], 0]]),
                    np.asarray([robust_trace[v[0], 1], robust_trace[v[1], 1]]),
                    np.asarray([index_list[v[0]], index_list[v[1]]]),
                    '-')

            # ax.plot(np.asarray([dcs_trace[v[0], 0], dcs_trace[v[1], 0]]),
            #         np.asarray([dcs_trace[v[0], 1], dcs_trace[v[1], 1]]),
            #         np.asarray([index_list[v[0]], index_list[v[1]]]),
            #         '--r')
    # ax.axis([0.0,15,-6,2.0,0,600)])
    ax.set_xlim(0.0, 15)
    ax.set_ylim(-6.0, 2.0)
    # ax.set_zlim(200, 550)




    plt.show()
