# -*- coding:utf-8 -*-
# Created by steve @ 17-11-17 上午9:46
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
    v_data = np.loadtxt('/home/steve/Data/II/17/vertex_all_data.csv', delimiter=',')

    '''
            id | time ax ay az wx wy wz mx my mz pressure| x  y  z  vx vy vz| qx qy qz qw
            0  |   1   2  3 4  5   6  7 8  9  10 11      | 12 13 14 15 16 17| 18 19 20 21
            1 + 11 + 6 + 4 = 22
        '''

    plt.figure()
    plt.grid()
    plt.title('acc')
    for i in range(3):
        plt.plot(v_data[:, i + 2], label=str(i))
    plt.plot(np.linalg.norm(v_data[:, 2:5], axis=1), label="norm")
    plt.legend()

    plt.figure()
    plt.grid()
    plt.title('mag')
    for i in range(3):
        plt.plot(v_data[:, i + 8], label=str(i))

    plt.plot(np.linalg.norm(v_data[:, 8:11], axis=1), label="norm")
    plt.legend()

    plt.figure()
    plt.grid()
    plt.title('src_distance of mag')

    dis_matrix = np.zeros([v_data.shape[0],v_data.shape[0]])

    for i in range(5,v_data.shape[0]-10):
        for j in range(i,v_data.shape[0]-10):
            dis_matrix[i,j] = (np.linalg.norm(v_data[i-5:i+5,8:10]-v_data[j-5:j+5,8:10]))
            dis_matrix[j,i] = dis_matrix[i,j]*1.0

    plt.imshow(dis_matrix)
    plt.colorbar()


    plt.figure()
    plt.grid()
    plt.title('norm dis')

    norm_dis_matrix = np.zeros([v_data.shape[0],v_data.shape[0]])

    for i in range(5, v_data.shape[0]-10):
        for j in range(i,v_data.shape[0]-10):
            norm_dis_matrix[i,j] = np.linalg.norm(np.linalg.norm(v_data[i-5:i+5,8:10],axis=1)-
                                                  np.linalg.norm(v_data[j-5:j+5,8:10],axis=1))

    plt.imshow(norm_dis_matrix)
    plt.colorbar()


    plt.show()
