# -*- coding:utf-8 -*-
# Created by steve @ 17-12-1 下午1:33
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
import matplotlib

from palettable.colorbrewer.sequential import *
import seaborn as sns
from publib import set_style

from MagPreprocess import MagPreprocess

import timeit
import time

if __name__ == '__main__':

    # set_style("poster")
    # sns.set(style='whitegrid')
    # sns.set_context("paper")

    # dir_list = [16, 17, 20, 28, 33, 34, 35]
    dir_list = [34]
    # dir_list = [28,35]
    # dir_list = [16,17,20,28]
    plt.figure()
    plot_rows = 2
    plot_cols = len(dir_list)
    # plt.ion()
    # plt.show()
    # plt.subplot()
    # plt.figure()
    for dir_i in range(len(dir_list)):
        dir_str = dir_list[dir_i]
        dir_name = '/home/steve/Data/II/' + str(dir_str) + "/"

        v_data = np.loadtxt(dir_name + 'vertex_all_data.csv', delimiter=',')

        '''
                id | time ax ay az wx wy wz mx my mz pressure| x  y  z  vx vy vz| qx qy qz qw
                0  |   1   2  3 4  5   6  7 8  9  10 11      | 12 13 14 15 16 17| 18 19 20 21
                1 + 11 + 6 + 4 = 22
            '''
        mDetector = MagPreprocess.MagDetector(v_data[:, 8:11],
                                              v_data[:, 2:5],
                                              v_data[:, 12:],
                                              v_data[:, 11])

        mDetector.Step2Length(False)
        # mDetector.GetFFTDis(20.0)
        # mDetector.MultiLayerNormFFt([30.0, 25.0, 20.0, 15.0, 10.0, 5.0])
        mDetector.GetDirectDis(20.0, False)
        mDetector.GetZValue(False)
        mDetector.ConvertMagAttitude()
        # mDetector.GetZFFtDis(20.0)
        # mDetector.MultiLayerNZFFt([30, 25, 20.0, 15.0, 10.0, 5.0])
        # mDetector.GetRelativeAttDis(50.0)
        mDetector.MultiLayerANZFFt([30.0, 25.0, 20.0, 15.0, 10.0, 5.0], False)

        # trace in x-o-y
        plt.subplot(plot_rows, plot_cols, dir_i + 1)
        plt.title(str(dir_str) + ' trace')
        plt.plot(v_data[:, 12], v_data[:, 13], '--*')
        # plt.grid()

        # distance of fft features
        plt.subplot(plot_rows, plot_cols, dir_i + plot_cols + 1)
        plt.title(str(dir_str) + ' dis_matrix')
        # mDetector.tmp_mnza_mat = np.where(mDetector.tmp_mnza_mat > 20, 20, mDetector.tmp_mnza_mat)
        plt.imshow((mDetector.tmp_mnza_mat))
        plt.colorbar()

        # direct euler distance
        # plt.subplot(plot_rows, plot_cols, dir_i + plot_cols * 2 + 1)
        # plt.title(str(dir_str) + ' src dis matrix')
        # plt.imshow(mDetector.tmp_src_mat)
        # plt.colorbar()
        # if dir_i is 0:
        # plt.show(block=False)
        # plt.draw()
        # plt.pause(0.001)

    #
    print('begin cv2')
    import cv2

    cv2.namedWindow('the')
    # cv2.namedWindow('the2')
    cv2.createTrackbar('threshold', 'the', 0, 500, lambda x: x)

    t_mat = mDetector.tmp_mnza_mat * 1.0
    while (True):
        t_v = float(cv2.getTrackbarPos('threshold', 'the')) / 10.0
        t = np.where(t_mat > t_v,
                     t_v,
                     t_mat)
        # t = cv2.cvtColor(t,cv2.CV_8S)
        plt.figure(2)
        plt.imshow(t)
        # plt.savefig('./ttt.png')
        # t_figure = cv2.imread('./ttt.png')
        cv2.imshow('the', t / t.max())
        # plt.show()
        # cv2.imshow('the', t)
        cv2.waitKey(10)
    # plt.pause(1000000000000)
    # plt.waitforbuttonpress()
    # plt.show()
