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
    index_list = np.linspace(0,dcs_trace.shape[0],dcs_trace.shape[0])

    robust_trace = np.loadtxt('./Robust/test.txt', delimiter=',')


    fig = plt.figure()

    ax1 = fig.add_subplot(131,projection='3d')
    ax1.plot(dcs_trace[:,0],dcs_trace[:,1],index_list,'-+')


    plt.show()

