# -*- coding:utf-8 -*-
# Created by steve @ 17-12-23 下午7:46
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
    for data_num in [ 20, 28, 33, 34]:
        t_auc = AucBuilder.AUCBuilder(str(data_num))
        t_auc.compute_ref_mat(is_show=False)
        t_auc.compute_all_auc(is_show=False)
        auc_list.append(t_auc)
    print(time.time()-start_auc_time)

    # begin plot
    for auc in auc_list:
        plt.figure(figsize=(7.5,7.5))
        plt.subplot(2,2,1)
        plt.title('(a)')
        plt.plot(auc.trace_path[:,0],auc.trace_path[:,1],'-*')
        plt.grid()
        plt.subplot(2,2,3)
        plt.title('(c)')
        plt.imshow(auc.src_mat[15:-15,15:-15])
        # plt.grid()
        plt.subplot(2,2,4)
        plt.title('(d)')
        plt.imshow(auc.mnza_mat[15:-15,15:-15])
        # plt.grid()
        plt.subplot(2,2,2)
        plt.title('(b)')
        plt.plot(auc.mnza_FPR, auc.mnza_TPR, '*', label='MultiLayerFFT')
        plt.plot(auc.src_FPR, auc.src_TPR, '*', label='Original')
        plt.legend()
        plt.grid()
        plt.axis([0,1,0,1])

        plt.savefig(auc.dir_name[:-1]+'data.jpg', dpi = 1000)
    plt.show()



