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
    v_data = np.loadtxt('/home/steve/Data/II/20/vertex_all_data.csv',delimiter=',')

    '''
            id | time ax ay az wx wy wz mx my mz pressure| x  y  z  vx vy vz| qx qy qz qw
            0  |   1   2  3 4  5   6  7 8  9  10 11      | 12 13 14 15 16 17| 18 19 20 21
            1 + 11 + 6 + 4 = 22
        '''

    plt.figure()
    plt.grid()
    plt.title('acc')
    for i in range(3):
        plt.plot(v_data[:,i+2],label=str(i))
    plt.plot(np.linalg.norm(v_data[:,2:5],axis=1),label="norm")
    plt.legend()

    plt.figure()
    plt.grid()
    plt.title('mag')
    for i in range(3):
        plt.plot(v_data[:,i+8],label=str(i))

    plt.plot(np.linalg.norm(v_data[:,8:11],axis=1),label="norm")
    plt.legend()


    plt.show()