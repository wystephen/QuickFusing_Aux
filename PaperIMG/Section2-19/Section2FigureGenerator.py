# -*- coding:utf-8 -*-
# Created by steve @ 17-12-14 上午10:04
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

import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy as sp
import numpy as np
import seaborn


from scipy.spatial.distance import pdist, squareform

if __name__ == '__main__':
    bi_mat = np.loadtxt('19bi_mat.data')
    mnza_mat = np.loadtxt('19mnza_mat.data')
    pairs_mat_index = np.loadtxt('19pairs_mat.data')
    pairs_mat = np.zeros_like(mnza_mat)
    pairs_mat[pairs_mat_index[:, 0].astype(dtype=np.int),
              pairs_mat_index[:, 1].astype(dtype=np.int)] = 220

    trace_imu = np.loadtxt('text_imu.txt', delimiter=',')
    trace_graph = np.loadtxt('test.txt', delimiter=',')


    ref_dis_mat = np.zeros([trace_graph.shape[0],trace_graph.shape[0]])

    ref_dis_mat = squareform(pdist(trace_graph))
    ref_dis_mat = np.where(ref_dis_mat<1.6,1.0,0.0)

    '''
    Display several feature mat and save to png files.
    '''
    if False:
        rows = 3
        cols = 1
        # plt.subplot(rows,cols, 2)
        plt.figure()
        plt.title('bi_mat')
        plt.imshow(bi_mat)
        plt.colorbar()
        plt.savefig('bi_mat.png',dpi=1000)

        # plt.subplot(rows,cols, 1)
        plt.figure()
        plt.title('feature_mat')
        plt.imshow(mnza_mat)
        plt.colorbar()
        plt.savefig('feature_mat.png',dpi=1000)

        # plt.subplot(rows,cols, 3)
        plt.figure()
        plt.title('pairs_mat')
        plt.imshow(pairs_mat)
        plt.colorbar()
        plt.savefig('pairs_mat.png',dpi=1000)

        # plt.savefig('three_mat.png',dpi=10000)

        plt.figure()
        plt.plot(trace_imu[:, 0], trace_imu[:, 1], '+-', label='trace_imu')
        plt.plot(trace_graph[:, 0], trace_graph[:, 1], '+-', label='trace graph')
        plt.grid()
        plt.legend()
        plt.savefig('ref_trace.png',dpi  = 1000)

        plt.figure()

        plt.imshow(ref_dis_mat)
        plt.title('ref distance mat')
        plt.colorbar()
        plt.savefig('ref_dis.png',dpi=1000)



    threshold_list = np.asarray(range(0,600,1),dtype=np.float)
    threshold_list = threshold_list/10.0
    # print(threshold_list)

    TPR = np.zeros(threshold_list.shape[0])
    FPR = np.zeros(threshold_list.shape[0])

    real_positive = float(ref_dis_mat[ref_dis_mat>0.5].shape[0])
    real_negative = float(ref_dis_mat[ref_dis_mat<0.5].shape[0])

    print('real positive:',real_positive,' real negative: ', real_negative)

    for i in range(2,threshold_list.shape[0]):
        x_list,y_list = np.where(mnza_mat<threshold_list[i])
        tmp = ref_dis_mat[x_list,y_list]

        tp = tmp[tmp>0.5].shape[0]
        fp = tmp[tmp<0.5].shape[0]

        TPR[i] = float(tp)/ real_positive
        FPR[i] = float(fp)/real_negative

    plt.figure()
    # plt.plot(FPR,TPR,'*')
    plt.plot(threshold_list,TPR,'--+',label='TPR')
    plt.plot(threshold_list,FPR,'-+',label='FPR')
    plt.legend()
    plt.grid()







    plt.show()
