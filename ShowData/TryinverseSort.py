# -*- coding:utf-8 -*-
# Created by steve @ 17-12-4 下午7:07
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

import seaborn as sns

from skimage import measure, color

import timeit
import time

change_flag = True


def is_changed(k):
    global change_flag
    change_flag = True


if __name__ == '__main__':
    sns.set('paper', 'white')

    start_time = time.time()
    dir_name = '/home/steve/Data/II/34/'
    dir_name = '/home/steve/Data/II/34/'

    ### key 16 17 20 ||| 28  30  (31)
    ##  33 34 35

    v_data = np.loadtxt(dir_name + 'vertex_all_data.csv', delimiter=',')

    '''
            id | time ax ay az wx wy wz mx my mz pressure| x  y  z  vx vy vz| qx qy qz qw
            0  |   1   2  3 4  5   6  7 8  9  10 11      | 12 13 14 15 16 17| 18 19 20 21
            1 + 11 + 6 + 4 = 22
        '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(v_data[:, 12], v_data[:, 13], v_data[:, 14], '-*', label='trace 3d \\alpha ')
    ax.legend()

    mDetector = MagPreprocess.MagDetector(v_data[:, 8:11],
                                          v_data[:, 2:5],
                                          v_data[:, 12:],
                                          v_data[:, 11])

    mDetector.Step2Length(False)
    # mDetector.GetFFTDis(20.0)
    # mDetector.MultiLayerNormFFt([30.0, 25.0, 20.0, 15.0, 10.0, 5.0])
    # mDetector.GetDirectDis(20.0)
    mDetector.GetZValue(False)
    mDetector.ConvertMagAttitude()
    # mDetector.GetZFFtDis(20.0)
    # mDetector.MultiLayerNZFFt([30, 25, 20.0, 15.0, 10.0, 5.0])
    # mDetector.GetRelativeAttDis(50.0)
    mDetector.MultiLayerANZFFt([30, 20.0, 10.0, 5.0])

    print('begin cv2')
    import cv2

    cv2.namedWindow('the')
    cv2.namedWindow('the2')
    cv2.namedWindow('the3')
    cv2.namedWindow('the4')

    cv2.createTrackbar('threshold', 'the', 100, 500, is_changed)
    cv2.createTrackbar('line_len', 'the', 2220, 2550, is_changed)
    cv2.createTrackbar('line_gap', 'the', 0, 2550, is_changed)
    cv2.createTrackbar('c_size', 'the', 0, 50, is_changed)
    cv2.createTrackbar('ero_size', 'the', 0, 50, is_changed)
    cv2.createTrackbar('ero_times', 'the', 0, 30, is_changed)

    # search parameters
    cv2.createTrackbar('detector_threshold', 'the', 109, 255, is_changed)
    cv2.createTrackbar('less_len', 'the', 5, 200, is_changed)
    cv2.createTrackbar('less_rate', 'the', 33, 100, is_changed)
    cv2.createTrackbar('less_k', 'the', 10, 100, is_changed)

    t_mat = mDetector.tmp_mnza_mat * 1.0
    while (True):
        if not change_flag:
            cv2.waitKey(10)
            continue
        else:

            t_v = float(cv2.getTrackbarPos('threshold', 'the')) / 10.0
            t = np.where(t_mat > t_v,
                         t_v,
                         t_mat)

            cv2.imshow('the', t)
            # t = cv2.cvtColor(t,cv2.CV_8S)
            plt.figure(2)
            plt.imshow(t)

            t = (t / t.max())
            kernel_size = cv2.getTrackbarPos('c_size', 'the')
            if kernel_size > 0:
                kernel = np.zeros([kernel_size * 2 + 1, kernel_size * 2 + 1])
                kernel[:, kernel_size] = 1.0 / float(kernel_size)
                kernel[kernel_size, :] = 1.0 / float(kernel_size)
                kernel[kernel_size, kernel_size] = 1.0
                t = cv2.filter2D(t, -1, kernel)

            eros_size = cv2.getTrackbarPos('ero_size', 'the')
            eros_times = cv2.getTrackbarPos('ero_times', 'the')
            ero_kernel = np.zeros([eros_size, eros_size], np.uint8)
            for i in range(ero_kernel.shape[0]):
                for j in range(ero_kernel.shape[1]):
                    if i == j:
                        ero_kernel[i, j] = 1
                    elif i + j == ero_kernel.shape[0]:
                        ero_kernel[i, j] = 1

            for i in range(int(eros_times)):
                t = cv2.dilate(t, ero_kernel, 1)

            t = t * 255
            t = t.astype(dtype=np.uint8)

            line_len = cv2.getTrackbarPos('line_len', 'the')
            line_gap = cv2.getTrackbarPos('line_gap', 'the')
            # t = cv2.cvtColor(t, cv2.COlor)
            # convert
            # lines = cv2.HoughLinesP(t, 1, np.pi / 180 * 5, line_len, line_gap)
            # # print(type(lines))
            # if type(lines) is type(np.array([0])):
            #     for line in lines:
            #         for x1, y1, x2, y2 in line:
            #             cv2.line(t, (x1, y1), (x2, y2), (0, 255, 0), 2)

            line_img = np.zeros_like(t)
            lines = cv2.HoughLines(t, 1, np.pi / 180.0 * 5.0, line_len)
            # print('lines ;', lines )
            if type(lines) is type(np.array([0])):
                lines1 = lines[:, 0, :]  # 提取为为二维
                for rho, theta in lines1[:]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(t, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 0), 1)
            # cv2.imshow('the2', line_img)
            #
            t = (t.astype(dtype=np.float) / t.astype(dtype=np.float).max() * 255).astype(dtype=np.uint8)

            '''
            Search detector ....
            '''
            d_threshold = cv2.getTrackbarPos('detector_threshold', 'the')
            d_less_len = cv2.getTrackbarPos('less_len', 'the')
            d_less_rate = cv2.getTrackbarPos('less_rate', 'the')
            d_less_rate = float(d_less_rate) / 10.0
            d_less_k = cv2.getTrackbarPos('less_k', 'the')
            d_less_rate = float(d_less_k / 10.0)

            bi_mat = np.zeros_like(t)

            cv2.threshold(t, d_threshold, 255, cv2.THRESH_BINARY_INV, dst=bi_mat)

            flag_mat = np.zeros_like(t)

            # flag_mat = bi_mat

            labels = measure.label(bi_mat, connectivity=2)

            # print('labels:',labels.shape,labels.max(),labels.min())
            '''
            Focus here the  important way to create image
            
            '''
            begin_plot = time.time()
            for l_index in range(labels.max()):
                # print(l_index, np.where(labels==l_index))
                x_list, y_list = np.where(labels == l_index)
                # print([x_list, y_list])

                x_val_range = float(max(x_list)-min(x_list))
                y_val_range = float(max(y_list)-min(y_list))

                if len(x_list) > d_less_len and \
                        float(len(x_list)) / d_less_rate < float(x_val_range+y_val_range)and \
                        (x_val_range / d_less_k < y_val_range < x_val_range * d_less_k) and \
                        x_val_range > d_less_len and y_val_range > d_less_len:
                    flag_mat[x_list, y_list] += 200
            end_plot = time.time()
            # line_segment_detector = cv2.createLineSegmentDetector()
            #
            # line_segs = line_segment_detector.detect(bi_mat)[0]
            #
            # segment_img = np.zeros_like(bi_mat)
            # segment_img = line_segment_detector.drawSegments(segment_img,line_segs)

            print('plot cost time :', end_plot - begin_plot)

            # for i in range(t.shape[0]):
            #     for j in range(i + 1, t.shape[1]):
            #         xoff, yoff = i, j
            #         # x_list = list()
            #         # y_list = list()
            #         point_list = list()
            #         while True:

            # cv2.imshow('the4',segment_img)
            cv2.imshow('the2', flag_mat)
            cv2.imshow('the3', bi_mat)
            cv2.imshow('the', t)
            # plt.show()
            # cv2.imshow('the', t)
            change_flag = False
            cv2.waitKey(10)
