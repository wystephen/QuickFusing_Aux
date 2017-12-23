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
import  numpy as np

import os
import  time


if __name__ == '__main__':
    dir_num_list = [19,28,33,34]
    for dir_num in dir_num_list:
        graph_trace = np.loadtxt('../'+str(dir_num)+'/test.txt',delimiter=',')
        imu_trace = np.loadtxt('../'+str(dir_num)+'/text_imu.txt',delimiter=',')
        pair = np.loadtxt('../'+str(dir_num)+'/pair.txt',delimiter=',')

        plt.figure()
        plt.plot(graph_trace[:,0],graph_trace[:,1],label='graph-optimized')
        plt.plot(imu_trace[:,0],imu_trace[:,1],label='zupt')
        plt.grid()
        plt.legend()
        plt.savefig(str(dir_num)+'trace.jpg',dpi=1000)
    plt.show()


