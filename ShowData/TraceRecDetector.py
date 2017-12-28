# -*- coding:utf-8 -*-
# Created by steve @ 17-12-23 上午10:32
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

import scipy as sp
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
import os
import seaborn


class TraceObject:
    def __init__(self, trace_3d):
        '''
        Find out shape of trace(a rectangle).

        :param trace_3d:
        '''
        self.trace_3d = trace_3d

    def find_rotation(self):
        '''
        find
        :return:
        '''




if __name__ == '__main__':
    trace = np.loadtxt('../PaperIMG/Section4-all/28/test.txt',
                       delimiter=',')

    # plt.figure()
    fig_trace = plt.figure()
    ax = fig_trace.add_subplot(111,projection='3d')

    plt.figure()
    plt.plot(trace[:, 0], trace[:, 1])
    plt.grid()


    ax.plot(trace[:,0],trace[:,1],trace[:,2],'*-')
    ax.grid()
    plt.show()
