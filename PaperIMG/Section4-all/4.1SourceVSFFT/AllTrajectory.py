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
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform

from MagPreprocess import MagPreprocess
import time

import os

from ShowData import AucBuilder, TraceProcess

if __name__ == '__main__':
    auc_list = list()
    start_auc_time = time.time()
    print()
    for data_num in [20, 28, 33, 34]:
        t_auc = AucBuilder.AUCBuilder(str(data_num))
        t_auc.compute_ref_mat(is_show=False)
        t_auc.compute_all_auc(is_show=False)
        auc_list.append(t_auc)
    print(time.time() - start_auc_time)

    # begin plot
    for auc in auc_list:
        fig = plt.figure(figsize=(7.5, 7.5))
        ax = fig.add_subplot(221, projection='3d')
        ax.set_title('(a)')
        trace_process = TraceProcess.TraceObject(auc.trace_path)
        ref_trace = trace_process.trace_normalized()
        index_list = np.linspace(0, ref_trace.shape[0], ref_trace.shape[0])
        ax.plot(ref_trace[:, 0], ref_trace[:, 1], index_list, '-*')
        ax.grid()
        ax = fig.add_subplot(2, 2, 3)
        ax.set_xlabel('x/m')
        ax.set_ylabel('y/m')
        ax.set_zlabel('z')



        ax.set_title('(c)')
        ax.imshow(auc.src_mat[15:-15, 15:-15])
        # plt.grid()
        ax = fig.add_subplot(2, 2, 4)
        ax.set_title('(d)')
        ax.imshow(auc.mnza_mat[15:-15, 15:-15])
        # plt.grid()
        ax = fig.add_subplot(2, 2, 2)
        ax.set_title('(b)')
        ax.plot(auc.mnza_FPR, auc.mnza_TPR, '*', label='MultiLayerFFT')
        ax.plot(auc.src_FPR, auc.src_TPR, '*', label='Original')
        ax.legend()
        ax.grid()
        ax.axis([0, 1, 0, 1])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

        fig.savefig(auc.dir_name[:-1] + 'data.jpg', dpi=1000)
    plt.show()
