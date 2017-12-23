# -*- coding:utf-8 -*-
# Created by steve @ 17-12-23 下午3:21
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
from scipy.spatial.distance import pdist, squareform

import os


class AUCBuilder(object):
    def __init__(self, dir_name):
        '''
        load data, according to a dir_name
        :param dir_name:
        '''
        if not dir_name[-1] is '/':
            dir_name = dir_name + '/'
        for file_name in os.listdir(dir_name):
            if 'bi_mat' in file_name:
                self.bi_mat = np.loadtxt(dir_name + file_name)

            if 'mnza_mat' in file_name:
                self.mnza_mat = np.loadtxt(dir_name + file_name)

            if 'pairs_mat' in file_name:
                self.pairs_mat = np.loadtxt(dir_name + file_name)

            if 'result_mat' in file_name:
                self.result_mat = np.loadtxt(dir_name + file_name)

            if 'source_distance_mat' in file_name:
                self.src_mat = np.loadtxt(dir_name + file_name)
                self.src_mat[:, :15] = np.max(self.src_mat)
                self.src_mat[:15, :] = np.max(self.src_mat)

            if 'pair.txt' in file_name:
                self.pair_map = np.loadtxt(dir_name + file_name, delimiter=',')

            if 'test' in file_name:
                self.trace_path = np.loadtxt(dir_name + file_name, delimiter=',')

    def compute_ref_mat(self, distance_threshold: float = 1.6, is_show: bool = True):
        '''
        Compute reference matrix according to the trajectory generated by graph optimization
        :param distance_threshold:
        :param is_show:
        :return:
        '''

        assert (self.mnza_mat.shape == self.src_mat.shape)
        self.ref_mat = np.zeros_like(self.mnza_mat)

        # for i in range(self.ref_mat.shape[0]):
        #     for j in range(self.ref_mat.shape[1]):
        self.ref_mat = squareform(pdist(self.trace_path[:, :2]))
        assert (self.mnza_mat.shape == self.ref_mat.shape)

        if is_show:
            plt.figure()
            plt.title('ref before threshold')
            plt.imshow(self.ref_mat)
            plt.colorbar()

        self.ref_mat = np.vectorize(lambda x: 1.0 if x < distance_threshold else 0.0)(self.ref_mat)

        if is_show:
            plt.figure()
            plt.title('ref bi')
            plt.imshow(self.ref_mat)

    def compute_all_auc(self, interval_num: int = 1000, is_show: bool = True):

        self.mnza_threshold_list = np.linspace(0.0, np.max(self.mnza_mat), interval_num)
        self.src_threshold_list = np.linspace(0.0, np.max(self.src_mat), interval_num)

        self.mnza_TPR, self.mnza_FPR = self.compute_auc(feature_mat=self.mnza_mat,
                                                        src_ref_mat=self.ref_mat,
                                                        threshold_list=self.mnza_threshold_list,
                                                        if_show=True)

        self.src_TPR, self.src_FPR = self.compute_auc(feature_mat=self.src_mat,
                                                      src_ref_mat=self.ref_mat,
                                                      threshold_list=self.src_threshold_list,
                                                      if_show=True)

        plt.figure()
        plt.title('AUC')
        plt.plot(self.mnza_FPR, self.mnza_TPR, '*', label='mnza')
        plt.plot(self.src_FPR, self.src_TPR, '*', label='src')
        plt.legend()
        plt.grid()
        plt.axis([0.0, 1.0, 0.0, 1.0])

    def compute_auc(self, *, feature_mat, src_ref_mat, threshold_list, if_show: bool = False):
        banned_ref_mat = src_ref_mat.copy()
        banned_feature_mat = feature_mat.copy()
        max_of_feature_mat = np.max(feature_mat) + 1.0
        banned_counter = 0

        for i in range(banned_ref_mat.shape[0]):
            for j in range(banned_ref_mat.shape[1]):
                if abs(i - j) < 5:
                    banned_ref_mat[i, j] = 0.0
                    banned_feature_mat[i, j] = max_of_feature_mat
                    banned_counter += 1.0

        real_positive = float(banned_ref_mat[banned_ref_mat > 0.5].shape[0])
        real_negative = float(banned_ref_mat[banned_ref_mat < 0.5].shape[0] - banned_counter)

        TPR = np.zeros_like(threshold_list)
        FPR = np.zeros_like(threshold_list)

        for i in range(threshold_list.shape[0]):
            x_l, y_l = np.where(banned_feature_mat < threshold_list[i])
            tmp = banned_ref_mat[x_l, y_l]

            tp = tmp[tmp > 0.5].shape[0]
            fp = tmp[tmp < 0.5].shape[0]

            TPR[i] = float(tp) / real_positive
            FPR[i] = float(fp) / real_negative

        return TPR.copy(), FPR.copy()

    def display_loaded_file(self, is_show=True):
        '''
        display resource data from __init__
        :param is_show: whether run plt.show() at the end of this function.
        :return:
        '''
        plt.figure()
        plt.imshow(self.mnza_mat)
        plt.title('mnza_mat')

        plt.figure()
        plt.imshow(self.src_mat)
        plt.title('src_mat')

        plt.figure()
        plt.plot(self.trace_path[:, 0], self.trace_path[:, 1])
        plt.grid()

        if is_show:
            plt.show()


if __name__ == '__main__':
    auc_b = AUCBuilder('./28')

    auc_b.compute_ref_mat()
    auc_b.compute_all_auc()

    auc_b.display_loaded_file()
