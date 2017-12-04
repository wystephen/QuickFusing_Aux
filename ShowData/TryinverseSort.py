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


import timeit
import time

if __name__ == '__main__':
    sns.set('paper','white')

    start_time = time.time()
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

    ax.plot(v_data[:, 12], v_data[:, 13], v_data[:, 14], '-*',label='trace 3d \\alpha ')
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
    mDetector.MultiLayerANZFFt([30.0, 25.0, 20.0, 15.0, 10.0, 5.0])

    print('begin cv2')
    import cv2

    cv2.namedWindow('the')
    # cv2.namedWindow('the2')
    cv2.createTrackbar('threshold', 'the', 0, 500, lambda x: x)
    cv2.createTrackbar('line_len','the',0,2550,lambda y:y)
    cv2.createTrackbar('line_gap','the',0,2550,lambda x:x)
    cv2.createTrackbar('c_size','the',1,50,lambda x:x)
    cv2.createTrackbar('ero_size','the',1,50,lambda x:x)


    t_mat = mDetector.tmp_mnza_mat * 1.0
    while (True):
        t_v = float(cv2.getTrackbarPos('threshold', 'the')) / 10.0
        t = np.where(t_mat > t_v,
                     t_v,
                     t_mat)
        # t = cv2.cvtColor(t,cv2.CV_8S)
        plt.figure(2)
        plt.imshow(t)


        t = (t / t.max())
        kernel_size = cv2.getTrackbarPos('c_size','the')
        kernel = np.zeros([kernel_size*2+1,kernel_size*2+1])
        kernel[:,kernel_size] = 1.0/float(kernel_size)
        kernel[kernel_size,:] = 1.0 / float(kernel_size)
        kernel[kernel_size,kernel_size] = 1.0
        t = cv2.filter2D(t,-1,kernel)

        eros_size = cv2.getTrackbarPos('ero_size','the')
        ero_kernel = np.zeros([eros_size,eros_size],np.uint8)
        for i in range(ero_kernel.shape[0]):
            for j in range(ero_kernel.shape[1]):
                if i==j:
                    ero_kernel[i,j] =1
                elif i+j == ero_kernel.shape[0]:
                    ero_kernel[i,j] = 1

        t = cv2.dilate(t,ero_kernel,1)


        t = t*255
        t = t.astype(dtype=np.uint8)



        line_len = cv2.getTrackbarPos('line_len','the')
        line_gap = cv2.getTrackbarPos('line_gap','the')
        # t = cv2.cvtColor(t, cv2.COlor)
        lines = cv2.HoughLinesP(t,1,np.pi/180*5,line_len,line_gap)
        # print(type(lines))
        if type(lines) is type(np.array([0])):
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(t,(x1,y1),(x2,y2),(0,255,0),2)


        #



        cv2.imshow('the', t )
        # plt.show()
        # cv2.imshow('the', t)
        cv2.waitKey(10)