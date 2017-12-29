# -*- coding:utf-8 -*-
# Created by steve @ 17-12-23 下午9:06
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

if __name__ == '__main__':
    dir_num_list = [19, 28, 33, 34]
    for dir_num in dir_num_list:
        graph_trace = np.loadtxt('../' + str(dir_num) + '/test.txt', delimiter=',')
        imu_trace = np.loadtxt('../' + str(dir_num) + '/text_imu.txt', delimiter=',')
        pair = np.loadtxt('../' + str(dir_num) + '/pair.txt', delimiter=',')
        if dir_num is dir_num_list[0]:
            rec_detector = TraceProcess.TraceObject(graph_trace)
            new_graph_trace = rec_detector.trace_normalized()
        else:
            new_graph_trace = rec_detector.trace_alig(graph_trace)

        # new_imu_trace = rec_detector.rotate_2d(imu_trace[:,:2], rec_detector.right_angle)
        new_imu_trace = rec_detector.trace_alig(imu_trace)

        plt.figure()
        plt.plot(new_graph_trace[:, 0], new_graph_trace[:, 1], label='graph-optimized')
        plt.plot(new_imu_trace[:, 0], new_imu_trace[:, 1], label='zupt')
        plt.grid()
        plt.legend()
        plt.xlabel('x/m')
        plt.ylabel('y/m')

        plt.savefig(str(dir_num) + 'trace.jpg', dpi=1000)
    plt.show()
