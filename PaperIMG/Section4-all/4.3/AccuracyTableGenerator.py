# -*- coding:utf-8 -*-
# Created by steve @ 18-1-12 上午8:52
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
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

import os
import time

from ShowData import TraceProcess
from ShowData import TraceCompare

if __name__ == '__main__':
    dir_num_list = [19, 20, 28, 33, 34]
    for dir_num in dir_num_list:
        graph_trace = np.loadtxt('../' + str(dir_num) + '/test.txt', delimiter=',')
        imu_trace = np.loadtxt('../' + str(dir_num) + '/text_imu.txt', delimiter=',')
        pair = np.loadtxt('../' + str(dir_num) + '/pair.txt', delimiter=',')
        ref_trace = np.loadtxt('../4.1SourceVSFFT/' + str(dir_num) + '/test.txt', delimiter=',')
        if dir_num is 20:
            ref_trace[:, 0] *= -1.0
        # plt.figure()
        # plt.plot(graph_trace[:, 0], graph_trace[:, 1], label='graph')
        # plt.plot(ref_trace[:, 0], ref_trace[:, 1], label='ref')
        # plt.legend()
        # plt.grid()
        # plt.title(str(dir_num))
        # print("size of ref:", ref_trace.shape[0], ' size of graph:', graph_trace.shape[0])
        tc = TraceCompare.TraceCompare(graph_trace, ref_trace)
        print('data:',dir_num,'error:',tc.error)



    plt.show()
