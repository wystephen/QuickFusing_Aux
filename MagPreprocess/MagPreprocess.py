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

from transforms3d.euler import euler2mat, mat2euler


class MagDetector:
    def __init__(self,
                 mag_data,
                 acc_data,
                 pose_data,
                 pressure):
        '''

        :param mag_data:
        :param acc_data:
        :param pose_data: x,y,z,vx,vy,vz,qx,qy,qz,qw
        :param pressure:
        '''
       # self.data = data
        self.mag_data = mag_data
        self.pose_data = pose_data
        self.pressure_data = pressure
        self.acc_data = acc_data

    def Step2Length(self):
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
                                      [:, 0], self.mag_norm, kind='linear')

        plt.figure()
        plt.title('inter')
        tx = np.linspace(0, self.length_array[-1],
                         self.length_array.shape[0] * 10.0)

        plt.plot(tx, self.f(tx), 'r+', label='interp')
        plt.plot(self.length_array, self.mag_norm, 'b*', label='source mag norm')
        plt.legend()
        plt.grid()

    def GetFFTDis(self, length, ifshow=True):

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

        self.tmp_fft_mat = np.zeros([self.mag_fft_feature.shape[0],
                                     self.mag_fft_feature.shape[0]])

        for i in range(self.mag_fft_feature.shape[0]):
            for j in range(i, self.mag_fft_feature.shape[0]):

                if np.linalg.norm(self.mag_fft_feature[i, :] - np.zeros([
                    1, len(test_shape_fft)], dtype=np.complex
                )) < 10.0:
                    continue

                if np.linalg.norm(self.mag_fft_feature[j, :] - np.zeros([
                    1, len(test_shape_fft)], dtype=np.complex
                )) < 10.0:
                    continue

                self.tmp_fft_mat[i, j] = np.linalg.norm(
                    (self.mag_fft_feature[i, :] - self.mag_fft_feature[j, :])
                )

                # if (self.tmp_fft_mat[i, j] > 4000):
                #     self.tmp_fft_mat[i, j] = 5000
                self.tmp_fft_mat[j, i] = self.tmp_fft_mat[i, j]

        if ifshow:
            plt.figure()
            plt.title('dis fft')

            plt.imshow(self.tmp_fft_mat)
            plt.colorbar()

            plt.figure()
            plt.title('gradient')
            tmp_grandient = self.tmp_fft_mat[:, 1:] - self.tmp_fft_mat[:, :-1]
            ttmp_grandient = tmp_grandient[:, 1:] - tmp_grandient[:, :-1]
            # for i in range(ttmp_grandient.shape[0]):
            #     for j in range(ttmp_grandient.shape[1]):
            #         if ttmp_grandient[i,j] < 5000:
            #             ttmp_grandient[i,j] = 0

            plt.imshow((ttmp_grandient))
            plt.colorbar()

    def MultiLayerFFt(self, layer_array, ifshow=True):
        print(layer_array)

        for index in range(len(layer_array)):
            self.GetFFTDis(layer_array[index], False)
            if index == 0:
                self.tmp_mul_mat = self.tmp_fft_mat
            else:
                self.tmp_mul_mat += self.tmp_fft_mat

        # for i in range(self.tmp_mul_mat.shape[0]):
        #     for j in range(self.tmp_mul_mat.shape[1]):
        #         if self.tmp_mul_mat[i, j] > 5000:
        #             self.tmp_mul_mat[i, j] = 5000.0

        if ifshow:
            plt.figure()
            plt.title('mul fft dis')
            plt.imshow(self.tmp_mul_mat)
            plt.colorbar()

    def GetDirectDis(self, length, ifshow=True):
        # print(length)

        intevel_length = 0.1

        self.mag_src_signal = np.zeros([self.length_array.shape[0],
                                        np.linspace(0, length, int(length / intevel_length)).shape[0]])

        # mag src singal ~
        for i in range(self.mag_src_signal.shape[0]):
            if self.length_array[i] < length / 2.0 or \
                            self.length_array[i] > self.length_array[-1] - length / 2.0:
                continue
            else:
                the_x = np.linspace(self.length_array[i] - length / 2.0,
                                    self.length_array[i] + length / 2.0,
                                    int(length / intevel_length))
                self.mag_src_signal[i, :] = self.f(the_x)

        self.tmp_src_mat = np.zeros([
            self.mag_src_signal.shape[0],
            self.mag_src_signal.shape[0]
        ])

        for i in range(self.mag_src_signal.shape[0]):
            for j in range(i, self.mag_src_signal.shape[0]):
                if self.length_array[i] < length / 2.0 or \
                                self.length_array[i] > self.length_array[-1] - length / 2.0:
                    continue
                else:
                    the_x = np.linspace(self.length_array[i] - length / 2.0,
                                        self.length_array[i] + length / 2.0,
                                        int(length / 0.1))
                    # self.mag_src_signal = self.f(the_x)
                    the_inv_x = np.linspace(self.length_array[i] + length / 2.0,
                                            self.length_array[i] - length / 2.0,
                                            int(length / 0.1))

                    self.tmp_src_mat[i, j] = np.min(
                        [np.linalg.norm(self.mag_src_signal[j] - self.f(the_x)),
                         np.linalg.norm(self.mag_src_signal[j] - self.f(the_inv_x))]
                    )
                    self.tmp_src_mat[j, i] = self.tmp_src_mat[i, j]

        if ifshow:
            plt.figure()
            plt.title('dis src')
            plt.imshow(self.tmp_src_mat)
            plt.colorbar()

    def GetZValue(self, ifshow=True):

        angle_all = np.zeros([self.acc_data.shape[0],2])

        for i in range()



        if ifshow:
            plt.figure()
            for
