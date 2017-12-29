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


class TraceObject(object):
    def __init__(self, trace_3d):
        '''
        Find out shape of trace(a rectangle).

        :param trace_3d:
        '''
        self.trace_3d = trace_3d

    def rotate_2d(self, trace, angle):
        '''
        rotate serious of points according to given angle(in radio)
        :param trace:
        :param angle:
        :return:
        '''
        tmp_trace = np.zeros_like(trace[:, :2])

        tmp_trace[:, 0] = self.trace_3d[:, 0] * np.cos(angle) - \
                          self.trace_3d[:, 1] * np.sin(angle)
        tmp_trace[:, 1] = self.trace_3d[:, 1] * np.cos(angle) + \
                          self.trace_3d[:, 0] * np.sin(angle)
        return tmp_trace

    def find_rotation(self):
        '''
        find
        :return:
        '''
        all_angle = np.linspace(-np.pi, np.pi, 3000)

        value_list = np.zeros([all_angle.shape[0], 4])
        tmp_trace = np.zeros_like(self.trace_3d[:, :2])
        for index, angle in enumerate(all_angle):
            tmp_trace = self.rotate_2d(self.trace_3d, angle)

            value_list[index, 0] = np.max(tmp_trace[:, 0]) - np.min(tmp_trace[:, 0])
            value_list[index, 1] = np.max(tmp_trace[:, 1]) - np.min(tmp_trace[:, 1])
            value_list[index, 2] = tmp_trace[0, 0] - np.min(tmp_trace[:, 0])
            value_list[index, 3] = tmp_trace[0, 1] - np.min(tmp_trace[:, 1])

        index_list = np.argsort(np.sum(value_list[:, 2:], axis=1))
        plt.figure()
        plt.plot(index_list, '+')
        plt.grid()
        self.right_angle = all_angle[index_list[0]]

        # for i in index_list:
        #     print(i,all_angle[i],value_list[i,2]+value_list[i,3])
        #     if (value_list[i, 0] > value_list[i, 1]) and \
        #             ((value_list[i, 2] + value_list[i, 3]) < 16):
        #         self.right_angle = all_angle[i]
        #         break

        plt.figure()
        plt.title('rotated trajectory')
        the_trace = self.rotate_2d(self.trace_3d, self.right_angle)
        plt.plot(the_trace[:, 0], the_trace[:, 1])
        plt.grid()

        plt.figure()
        plt.plot(all_angle, value_list[:, 0], label='x')
        plt.plot(all_angle, value_list[:, 1], label='y')
        plt.plot(all_angle, value_list[:, 0] + value_list[:, 1], label='sum')
        plt.grid()
        plt.legend()

        # plt.figure()
        plt.plot(all_angle,value_list[:, 2], label='x offset')
        plt.plot(all_angle, value_list[:, 3], label='y offset')
        plt.plot(all_angle, value_list[:, 2] + value_list[:, 3], label='norm')
        plt.grid()
        plt.legend()


if __name__ == '__main__':
    trace = np.loadtxt('../PaperIMG/Section4-all/28/test.txt',
                       delimiter=',')

    to = TraceObject(trace)
    to.find_rotation()

    # plt.figure()
    fig_trace = plt.figure()
    ax = fig_trace.add_subplot(111, projection='3d')

    plt.figure()
    plt.plot(trace[:, 0], trace[:, 1])
    plt.grid()

    ax.plot(trace[:, 0], trace[:, 1], trace[:, 2], '*-')
    ax.grid()
    plt.show()
