# -*- coding:utf-8 -*-
# Created by steve @ 17-11-22 上午11:09
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

import scipy as sp
import numpy as np

import matplotlib.pylab as plt

from scipy import interpolate
from scipy.fftpack import fft, ifft


class MagDetector:
    def __init__(self,
                 mag_data,
                 pose_data,
                 pressure):
        '''

        :param mag_data:
        :param pose_data:
        :param pressure:
        '''
        # self.data = data
        self.mag_data = mag_data
        self.pose_data = pose_data
        self.pressure_data = pressure

    def Step2Length(self, length):
        '''

        :param length:
        :return:
        '''

        self.length_array = np.zeros([self.mag_data.shape[0], 1])
        self.mag_norm = np.linalg.norm(self.mag_data, axis=1)

        for i in range(1, self.length_array.shape[0]):
            self.length_array[i] = self.length_array[i - 1] + \
                                   np.linalg.norm(self.pose_data[i, :]
                                                  - self.pose_data[i - 1, :])

        self.f = interpolate.interp1d(self.length_array
                                      [:, 0], self.mag_norm, kind='cubic')

        tx = np.linspace(0.0, self.length_array[-1], num=self.length_array.shape[0] * 10)

        # self.mag_fft_list = list(self.length_array.shape[0])
        test_shape_fft = fft(np.linspace(0, length, int(length / 0.5)))
        self.mag_fft_feature = np.zeros([self.length_array.shape[0],
                                         len(test_shape_fft)],
                                        dtype=np.complex)

        for i in range(0, self.length_array.shape[0]):

            if self.length_array[i] < length / 2.0 or \
                            self.length_array[i] > self.length_array[-1] - length / 2.0:
                continue
            else:
                the_x = np.linspace(self.length_array[i] - length / 2.0,
                                    self.length_array[i] + length / 2.0,
                                    int(length / 0.5))
                yyt = fft(self.f(the_x))
                self.mag_fft_feature[i, :] = yyt
                # print(i, yyt, yyt.real, yyt.imag)

        plt.figure()
        plt.title('dis fft')

        self.tmp_fft_mat = np.zeros([self.mag_fft_feature.shape[0],
                                     self.mag_fft_feature.shape[0]]) + 10000000000

        for i in range(self.mag_fft_feature.shape[0]):
            for j in range(i, self.mag_fft_feature.shape[0]):

                if np.linalg.norm(self.mag_fft_feature[i, :] - np.zeros([
                    1, len(test_shape_fft)], dtype=np.complex
                )) < 10.0:
                    continue

                self.tmp_fft_mat[i, j] = np.linalg.norm(
                    self.mag_fft_feature[i, :] - self.mag_fft_feature[j, :]
                )

                # if (self.tmp_fft_mat[i, j] > 4000):
                #     self.tmp_fft_mat[i, j] = 5000
                self.tmp_fft_mat[j, i] = self.tmp_fft_mat[i, j]

        plt.imshow(self.tmp_fft_mat)
        plt.colorbar()

        plt.figure()
        plt.title('inter')

        plt.plot(tx, self.f(tx), 'r+', label='interp')
        plt.plot(self.length_array, self.mag_norm, 'b*', label='source mag norm')
        plt.legend()
        plt.grid()
