# -*- coding:utf-8 -*-
# Created by steve @ 17-12-24 下午1:46
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

import  numpy as np
import scipy as sp
import matplotlib.pyplot as plt


if __name__ == '__main__':
    pair_mat = np.loadtxt('./DCS/pair.txt',delimiter=',')

    dcs_trace = np.loadtxt('./DCS/test.txt',delimiter=',')

    robust_trace = np.loadtxt('./Robust/test.txt',delimiter=',')

    plt.figure()
    plt.subplot(1,2,1)
    plt.grid()
    plt.plot(dcs_trace[:,0],dcs_trace[:,1],label='path')
    for i, v in enumerate(pair_mat):
        v = pair_mat[i,:].astype(dtype=np.int)
        plt.plot([dcs_trace[v[0],0],dcs_trace[v[1],0]],
                 [dcs_trace[v[0],0],dcs_trace[v[1],1]],
                 '--g',label='pairs')
    plt.legend()

    plt.subplot(1,2,2)
    plt.grid()
    plt.plot(robust_trace[:,0],robust_trace[:,1],label='paht')
    for i in range(pair_mat.shape[0]):
        v=pair_mat[i,:].astype(dtype=np.int)
        plt.plot([robust_trace[v[0],0],robust_trace[v[1],0]],
                 [robust_trace[v[0],1],robust_trace[v[1],1]],
                 '--g',label='pairs')
    plt.legend()
    plt.show()


