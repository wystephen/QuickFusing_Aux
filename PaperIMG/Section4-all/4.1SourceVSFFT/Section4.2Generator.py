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
        plt.figure(figsize=(8.0, 4.0))
        plt.subplot(1, 3, 1)
        plt.title('(a)')
        plt.imshow(auc.ref_mat[15:-15, 15:-15])
        plt.subplot(1, 3, 2)
        plt.title('(b)')
        plt.imshow(auc.bi_mat[15:-15, 15:-15])
        plt.subplot(1, 3, 2)
        plt.title('(c)')
        plt.imshow(auc.result_mat[15:-15, 15:-15])

        plt.savefig(auc.dir_name[:-1] + '4-2.jpg', dpi=1000)
    plt.show()
