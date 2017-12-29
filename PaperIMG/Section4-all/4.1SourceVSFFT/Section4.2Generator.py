# -*- coding:utf-8 -*-
# Created by steve @ 17-12-23 下午8:44
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

from MagPreprocess import MagPreprocess
import time

import os

from ShowData import AucBuilder

if __name__ == '__main__':
    auc_list = list()
    start_auc_time = time.time()
    print()
    for data_num in [20, 28, 33, 34]:
        t_auc = AucBuilder.AUCBuilder(str(data_num))
        t_auc.compute_ref_mat(distance_threshold=2.0, is_show=False)
        auc_list.append(t_auc)
    print(time.time() - start_auc_time)

    for index, auc in enumerate(auc_list):
        plt.figure(figsize=(9.0, 4.0))
        plt.subplot(1, 3, 1)
        plt.title('(a)')
        plt.xlabel('index')
        plt.ylabel('index')
        plt.imshow(auc.ref_mat[15:-15, 15:-15])
        plt.subplot(1, 3, 2)
        plt.title('(b)')
        plt.imshow(auc.bi_mat[15:-15, 15:-15])
        plt.xlabel('index')
        # plt.ylabel('index')
        ax = plt.subplot(1, 3, 3)
        ax.set_title('(c)')
        ax.set_xlabel('index')
        # plt.ylabel('index')
        for i in range(auc.result_mat.shape[0]):
            for j in range(auc.result_mat.shape[1]):
                if i == j:
                    auc.result_mat[i, j] = 200.0
        plt.imshow(auc.result_mat[15:-15, 15:-15])

        # x_list, y_list = np.where(auc.result_mat[15:-15, 15:-15] > 0.5)
        # ax.scatter(x_list, y_list)
        # ax.axis('equal')
        ax.set_xlim(0,auc.result_mat.shape[0]-30)
        ax.set_ylim(auc.result_mat.shape[0]-30,0)
        # plt.axis([0, auc.result_mat.shape[0] - 30,0, auc.result_mat.shape[0] - 30])
        # plt.xlim(0,auc.result_mat.shape[0]-30)
        # plt.ylim(0,auc.result_mat.shape[0]-30)

        plt.savefig(auc.dir_name[:-1] + '4-2.jpg', dpi=1000)
    plt.show()
