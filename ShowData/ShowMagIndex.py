# -*- coding:utf-8 -*-
# Created by steve @ 17-11-18 下午4:41
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

if __name__ == '__main__':
    v_data = np.loadtxt('/home/steve/Data/II/28/vertex_all_data.csv', delimiter=',')

    mag_edge = np.loadtxt('/home/steve/Code/QuickFusing/ResultData/pair.txt', delimiter=',').astype(dtype=np.int)

    flag_mat = np.zeros([v_data.shape[0], v_data.shape[0]])

    for i in range(mag_edge.shape[0]):
        flag_mat[mag_edge[i, 0], mag_edge[i, 1]] = 1.0

    plt.figure()
    plt.title('added mag constraint')

    plt.imshow(flag_mat)
    plt.colorbar()

    plt.show()
