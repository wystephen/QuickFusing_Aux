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

from mpl_toolkits.mplot3d import Axes3D

import os
import time

from ShowData import TraceProcess
from ShowData import TraceCompare

if __name__ == '__main__':
    dir_num_list = [19, 20, 28, 33, 34]
    for dir_num in dir_num_list:
        graph_trace = np.loadtxt('../' + str(dir_num) + '/test.txt', delimiter=',')
        imu_trace = np.loadtxt('../' + str(dir_num) + '/text_imu.txt', delimiter=',')
        # pair = np.loadtxt('../' + str(dir_num) + '/pair.txt', delimiter=',')
        pair = np.loadtxt('../4.1SourceVSFFT/'+str(dir_num)+'/'+str(dir_num)+'pairs_mat.data')
        ref_trace = np.loadtxt('../4.1SourceVSFFT/' + str(dir_num) + '/test.txt', delimiter=',')
        if dir_num is 20:
            ref_trace[:, 0] *= -1.0

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(graph_trace[:, 0], graph_trace[:, 1], graph_trace[:, 2], label='graph')
        # ax.plot(ref_trace[:, 0], ref_trace[:, 1], ref_trace[:, 2], label='ref')
        # ax.legend()
        # ax.grid()
        # ax.grid()
        # plt.figure()
        # plt.plot(graph_trace[:, 0], graph_trace[:, 1], label='graph')
        # plt.plot(ref_trace[:, 0], ref_trace[:, 1], label='ref')
        # plt.legend()
        # plt.grid()
        # plt.title(str(dir_num))
        # print("size of ref:", ref_trace.shape[0], ' size of graph:', graph_trace.shape[0])
        # print(np.min(pair), np.max(pair))
        min_pair = 1000000
        max_pair = 0
        t = np.zeros([graph_trace.shape[0], graph_trace.shape[0]])
        for i in range(pair.shape[0]):
            t[int(pair[i, 0]),int( pair[i, 1])] = 1.0
            if ((pair[i, 1] - pair[i, 0]) > 15) and \
                    15 < pair[i, 0] < graph_trace.shape[0] - 15 and \
                    15 < pair[i, 1] < graph_trace.shape[0] - 15:
                min_pair = min(min_pair, np.min(pair[i, :]))
                max_pair = max(max_pair, np.max(pair[i, :]))
        # plt.figure()
        # plt.imshow(t)
        min_pair = int(min_pair)
        max_pair = int(max_pair)
        print('min:', min_pair, 'max:', max_pair, 'size:', graph_trace.shape[0])

        tc = TraceCompare.TraceCompare(graph_trace, ref_trace)
        print('data:', dir_num, 'error:', tc.error)


        tc2 = TraceCompare.TraceCompare(graph_trace[min_pair:max_pair,:],ref_trace[min_pair:max_pair,:])
        print('data',dir_num,'paired error:',tc2.error)

    plt.show()
