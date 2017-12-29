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

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == '__main__':
    pair_mat = np.loadtxt('./DCS/corner_pair.txt', delimiter=',')
    pair_mat = pair_mat.astype(dtype=np.int)

    dcs_trace = np.loadtxt('./DCS/test.txt', delimiter=',')

    imu_trace = np.loadtxt('./DCS/text_imu.txt', delimiter=',')

    robust_trace = np.loadtxt('./Robust/test.txt', delimiter=',')

    # index_offset_list = np.linspace(0,100.0,dcs_trace.shape[0])
    # dcs_trace[:,0] += index_offset_list
    # robust_trace[:,0] += index_offset_list
    # dcs_trace[:,0] *= -1.0
    # robust_trace[:,0] *= -1.0

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.title('(a)')
    plt.grid()
    plt.plot(dcs_trace[:, 0], dcs_trace[:, 1], '-+', label='path')
    for i in range(pair_mat.shape[0]):
        v = pair_mat[i, :]
        plt.plot(np.asarray([dcs_trace[v[0], 0], dcs_trace[v[1], 0]]),
                 np.asarray([dcs_trace[v[0], 1], dcs_trace[v[1], 1]]),
                 '-r')
    plt.legend()

    fig = plt.subplot(1, 3, 2)
    fig.set_title('(b)')
    fig.grid()
    fig.plot(robust_trace[:, 0], robust_trace[:, 1], '-+', label='path')
    for i in range(pair_mat.shape[0]):
        v = pair_mat[i, :]
        fig.plot(np.asarray([robust_trace[v[0], 0], robust_trace[v[1], 0]]),
                 np.asarray([robust_trace[v[0], 1], robust_trace[v[1], 1]]),
                 '-r')

    # t_rectangle = plt.Rectangle([0.0, -6], width=15, height=8)
    # plt.add_path(t_rectangle.getpath())
    fig.add_patch(
        patches.Rectangle(
            (0.0, -6),
            15, 8, fill=False
        )
    )

    plt.legend()
    plt.subplot(1, 3, 3)
    plt.title('(c)')
    plt.grid()
    plt.plot(robust_trace[:, 0], robust_trace[:, 1], '-+', label='path')
    for i in range(pair_mat.shape[0]):
        v = pair_mat[i, :]
        plt.plot(np.asarray([robust_trace[v[0], 0], robust_trace[v[1], 0]]),
                 np.asarray([robust_trace[v[0], 1], robust_trace[v[1], 1]]),
                 '-r')
    plt.axis([0.0, 15, -6, 2.0])

    plt.savefig('compare_fig.jpg', dpi=1000)

    plt.figure()
    plt.plot(np.linalg.norm(robust_trace[1:, :2] - robust_trace[:-1, :2], axis=1) - np.linalg.norm(
        imu_trace[1:, :2] - imu_trace[:-1, :2], axis=1), '-+', label='robust')
    plt.plot(np.linalg.norm(dcs_trace[1:, :2] - dcs_trace[:-1, :2], axis=1) - np.linalg.norm(
        imu_trace[1:, :2] - imu_trace[:-1, :2], axis=1), '-+', label='dcs')
    # plt.plot(np.linalg.norm(imu_trace[1:,:2]-imu_trace[:-1,:2],axis=1),'-+',label='imu')
    plt.grid()
    plt.legend()

    robust_error = np.linalg.norm(robust_trace[1:, :2] - robust_trace[:-1, :2], axis=1) - np.linalg.norm(
        imu_trace[1:, :2] - imu_trace[:-1, :2], axis=1)
    dcs_error = np.linalg.norm(dcs_trace[1:, :2] - dcs_trace[:-1, :2], axis=1) - np.linalg.norm(
        imu_trace[1:, :2] - imu_trace[:-1, :2], axis=1)
    for i in range(1,robust_trace.shape[0]-1):
        robust_error[i] += robust_error[i-1]
        dcs_error[i] += dcs_error[i-1]

    plt.figure()
    plt.plot(robust_error,label='robust')
    plt.plot(dcs_error,label='dcs')
    plt.grid()
    plt.legend()



    plt.show()
