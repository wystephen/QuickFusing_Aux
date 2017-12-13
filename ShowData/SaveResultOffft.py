# -*- coding:utf-8 -*-
# Created by steve @ 17-12-11 上午9:13
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

from MagPreprocess import MagPreprocess, ransac

import seaborn as sns

from skimage import measure, color
from sklearn import linear_model, datasets

import timeit
import time

from multiprocessing import Pool

change_flag = True
segment_img = np.zeros([10, 10])


def is_changed(k):
    global change_flag
    change_flag = True


if __name__ == '__main__':
    sns.set('paper', 'white')

    start_time = time.time()
    data_dir = '/home/steve/Data/II/'
    data_num = 31
    # 16,17,19,20,28,31,32,33,34
    result_dir = '/home/steve/Data/II/result/' + str(data_num)
    dir_name = '/home/steve/Data/II/' + str(data_num) + '/'
    # dir_name = '/home/steve/Data/II/20/'

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
    start = time.time()
    mDetector.GetDirectDis(30.0)
    end = time.time()
    print('GetDirectDis time', end - start)
    mDetector.GetZValue(False)
    mDetector.ConvertMagAttitude()
    # mDetector.GetZFFtDis(20.0)
    # mDetector.MultiLayerNZFFt([30, 25, 20.0, 15.0, 10.0, 5.0])
    # mDetector.GetRelativeAttDis(50.0)
    start = time.time()
    mDetector.MultiLayerANZFFt([30, 25, 20.0, 15, 10.0, 5.0])
    end = time.time()
    print('MultiLayerANZFFT time:', end - start)

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
    cv2.createTrackbar('less_len', 'the', 6, 200, is_changed)
    cv2.createTrackbar('less_rate', 'the', 33, 100, is_changed)
    cv2.createTrackbar('less_k', 'the', 32, 100, is_changed)
    cv2.createTrackbar('max_r_error', 'the', 5, 100, is_changed)

    t_mat = mDetector.tmp_mnza_mat * 1.0
    while (True):
        if not change_flag:
            cv2.waitKey(1)
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
            d_less_k = float(d_less_k / 10.0)
            d_max_r_error = cv2.getTrackbarPos('max_r_error', 'the')
            d_max_r_error = float(d_max_r_error) / 10.0

            bi_mat = np.zeros_like(t)

            cv2.threshold(t, d_threshold, 255, cv2.THRESH_BINARY_INV, dst=bi_mat)

            labels = measure.label(bi_mat, connectivity=2)
            real_bi_mat = bi_mat.copy()

            bi_mat = cv2.cvtColor(bi_mat, cv2.COLOR_GRAY2RGB)
            # print('type bi mat', type(bi_mat), bi_mat.shape)

            '''
            Focus here the  important way to create image

            '''
            begin_plot = time.time()
            segment_img = np.zeros_like(t)
            segment_img_list = list()

            score_list = list()
            score_rel_list = list()

            # def process(l_index):
            for l_index in range(labels.max()):
                # nonlocal segment_img
                # global segment_img
                # global segment_img_list

                x_list, y_list = np.where(labels == l_index)
                x_val_range = float(max(x_list) - min(x_list))
                y_val_range = float(max(y_list) - min(y_list))

                if len(x_list) > d_less_len and \
                        float(len(x_list)) / d_less_rate < float(x_val_range + y_val_range) and \
                        (x_val_range / d_less_k < y_val_range < x_val_range * d_less_k) and \
                        x_val_range > d_less_len and y_val_range > d_less_len:

                    try:
                        ransac_line = linear_model.RANSACRegressor()
                        ransac_line.fit(x_list.reshape(-1, 1), y_list)
                        pre_y_list = ransac_line.predict(x_list.reshape(-1, 1))
                        the_tmp_score = np.linalg.norm(y_list - ransac_line.predict(x_list.reshape(-1, 1)))
                        score_list.append(the_tmp_score)
                        score_rel_list.append(the_tmp_score / float(len(x_list)))
                        x_distance = mDetector.length_array[max(x_list), 0] - mDetector.length_array[min(x_list), 0]
                        y_distance = mDetector.length_array[max(pre_y_list.astype(dtype=np.int)), 0] - \
                                     mDetector.length_array[min(pre_y_list.astype(dtype=np.int)), 0]

                        if the_tmp_score / float(len(x_list)) < d_max_r_error and \
                                0.8 < (y_distance / x_distance) < 1.20:
                            segment_img[x_list.astype(dtype=np.int),
                                        ransac_line.predict(x_list.reshape(-1, 1)).astype(dtype=np.int)] = 200

                            # plot~
                            bi_mat[x_list.astype(dtype=np.int),
                                   ransac_line.predict(x_list.reshape(-1, 1)).astype(dtype=np.int), 1] = 100
                            bi_mat[x_list.astype(dtype=np.int),
                                   ransac_line.predict(x_list.reshape(-1, 1)).astype(dtype=np.int), 0] = 200
                            bi_mat[x_list.astype(dtype=np.int),
                                   ransac_line.predict(x_list.reshape(-1, 1)).astype(dtype=np.int), 2] = 0

                    except ValueError:
                        print('some error here', l_index)

            # p = Pool()
            # the_range_list = range(labels.max())
            # map(process, the_range_list)
            # p.close()
            # p.join()
            # if len(score_list) > 0:
            # print('scorlist:', min(score_list), max(score_list))
            # print('scorlist:', min(score_rel_list), max(score_rel_list))

            end_plot = time.time()

            # print('plot cost time :', end_plot - begin_plot)

            cv2.imshow('the4', segment_img)
            # cv2.imshow('the2', flag_mat)
            cv2.imshow('the3', bi_mat)
            cv2.imshow('the', t)
            # plt.show()
            # cv2.imshow('the', t)
            change_flag = False
            cv2.waitKey(10)

            # save data to file
            # 1. write parameters to file
            import json

            data = dict()
            data['threshold'] = cv2.getTrackbarPos('threshold', 'the')
            data['line_len'] = cv2.getTrackbarPos('line_len', 'the')
            data['line_gap'] = cv2.getTrackbarPos('line_gap', 'the')
            data['c_size'] = cv2.getTrackbarPos('c_size', 'the')
            data['ero_size'] = cv2.getTrackbarPos('ero_size', 'the')
            data['ero_times'] = cv2.getTrackbarPos('ero_times', 'the')
            data['detector_threshold'] = cv2.getTrackbarPos('detector_threshold', 'the')
            data['less_len'] = cv2.getTrackbarPos('less_len', 'the')
            data['less_rate'] = cv2.getTrackbarPos('less_rate', 'the')
            data['less_k'] = cv2.getTrackbarPos('less_k', 'the')
            data['max_r_error'] = cv2.getTrackbarPos('max_r_error', 'the')

            p_f = open(result_dir + 'para.json', 'w')
            p_f.write(json.dumps(data))


            # 2. write data to file
            np.savetxt(result_dir+'source_distance_mat.data',mDetector.tmp_src_mat)
            np.savetxt(result_dir+'mnza_mat.data', mDetector.tmp_mnza_mat)
            np.savetxt(result_dir+'bi_mat.data', real_bi_mat)
            np.savetxt(result_dir+'result_mat.data',segment_img)

            pairs1,pairs2 = np.where(segment_img>100)
            pairs_mat = np.zeros([pairs1.shape[0],2])
            pairs_mat[:,0] = pairs1
            pairs_mat[:,1] = pairs2
            np.savetxt(result_dir+'pairs_mat.data', pairs_mat)

            np.savetxt(dir_name+'pairs.csv', pairs_mat) # save to data directory for graph optimization


            # 3. save image
