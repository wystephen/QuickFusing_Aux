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
from scipy.spatial.distance import pdist, squareform
import numexpr as ne

from transforms3d.euler import euler2mat, mat2euler, quat2axangle, quat2mat, quat2euler


# def dist(mat, i, j):
#     return np.linalg.norm(mat[i, :] - mat[j, :])


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

    def Step2Length(self, ifshow=True):
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
                                      [:, 0], self.mag_norm / self.mag_norm.mean(), kind='linear')

        if ifshow:
            plt.figure()
            plt.title('inter')
            tx = np.linspace(0, self.length_array[-1],
                             self.length_array.shape[0] * 10.0)

            plt.plot(tx, self.f(tx), 'r+', label='interp')
            plt.plot(self.length_array, self.mag_norm, 'b*', label='source mag norm')
            plt.legend()
            plt.grid()

    def ComputeDistanceFeatureSpace(self, feature_mat, ifshow=True):
        '''
        compute distance in feature space according to feature matrix
        :param feature_mat:  feature matrix
        :param ifshow: invalid {not used}
        :return:distance matrix
        '''
        dis_mat = np.zeros([feature_mat.shape[0], feature_mat.shape[0]])

        # dist  = lambda mat,i,j:return np.linalg.norm(mat[i,:]-mat[j,:])

        # for i in range(feature_mat.shape[0]):
        #     for j in range(i, feature_mat.shape[0]):
        #         dis_mat[i, j] = np.linalg.norm(
        #             feature_mat[i, :] - feature_mat[j, :]
        #         )
        #         dis_mat[j, i] = dis_mat[i, j]

        # for feature_mat, feature_mat, dis_mat in it:
        #     dis_mat[...]
        # dis_mat = np.linalg.norm(feature_mat[:, None] - feature_mat, axis=-1)
        # sp.spatial.d
        # print('feature mat shape', feature_mat.shape)
        dis_mat = squareform(pdist(feature_mat))

        return dis_mat

    def GetNormFFTDis(self, length, ifshow=True):

        tx = np.linspace(0.0, self.length_array[-1], num=self.length_array.shape[0] * 10)

        # self.mag_fft_list = list(self.length_array.shape[0])
        test_shape_fft = fft(np.linspace(0, length, int(length / 0.5)))
        self.mag_fft_feature = np.zeros([self.length_array.shape[0],
                                         len(test_shape_fft) ],
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

        self.tmp_fft_mat = self.ComputeDistanceFeatureSpace(self.mag_fft_feature)

        if ifshow:
            plt.figure()
            plt.title('dis fft')

            plt.imshow(self.tmp_fft_mat)
            plt.colorbar()

            plt.figure()
            plt.title('gradient')
            tmp_grandient = self.tmp_fft_mat[:, 1:] - self.tmp_fft_mat[:, :-1]
            ttmp_grandient = tmp_grandient[:, 1:] - tmp_grandient[:, :-1]

            plt.imshow((ttmp_grandient))
            plt.colorbar()

        return self.tmp_fft_mat

    def MultiLayerNormFFt(self, layer_array, ifshow=True):
        print(layer_array)

        for index in range(len(layer_array)):
            t_mat = self.GetNormFFTDis(layer_array[index], False)
            if index == 0:
                self.tmp_mul_mat = t_mat
            else:
                self.tmp_mul_mat += t_mat

        if ifshow:
            plt.figure()
            plt.title('mul fft dis')
            plt.imshow(self.tmp_mul_mat)
            plt.colorbar()

    def MultiLayerNZFFt(self, layer_array, ifshow=True):
        print('MultiLayerNZFFT', layer_array)

        for index in range(len(layer_array)):
            t_mat = self.GetNormFFTDis(layer_array[index], False)
            if index == 0:
                self.tmp_mnz_mat = t_mat
            else:
                self.tmp_mnz_mat += t_mat

        for index in range(len(layer_array)):
            self.tmp_mnz_mat += self.GetZFFtDis(layer_array[index], False)

        if ifshow:
            plt.figure()
            plt.title('mnz fft dis')
            plt.imshow(self.tmp_mnz_mat)
            plt.colorbar()

    def MultiLayerANZFFt(self, layer_array, ifshow=True):
        print('MultiLayerANZFFT', layer_array)

        for index in range(len(layer_array)):
            t_mat = self.GetNormFFTDis(layer_array[index], False)
            t_mat = t_mat / t_mat.mean()
            if index == 0:
                self.tmp_mnza_mat = t_mat
            else:
                self.tmp_mnza_mat += t_mat

        for index in range(len(layer_array)):
            t_mat = self.GetZFFtDis(layer_array[index], False)
            t_mat = t_mat / t_mat.mean()
            self.tmp_mnza_mat += t_mat

        for index in range(len(layer_array)):
            t_mat = self.GetRelativeAttDis(layer_array[index], False)
            t_mat = t_mat / t_mat.mean()
            self.tmp_mnza_mat += t_mat

        if ifshow:
            plt.figure()
            plt.title('mnz and att fft dis')
            plt.imshow(self.tmp_mnza_mat)
            plt.colorbar()

    def GetDirectDis(self, length, ifshow=True):
        '''

        :param length:
        :param ifshow:
        :return:
        '''
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

        angle_all = np.zeros([self.acc_data.shape[0], 2])
        tmp_acc_data = np.zeros_like(self.acc_data)
        self.convert_mag_data = np.zeros_like(self.mag_data)
        self.angle = np.zeros([self.mag_data.shape[0], 1])
        self.direct_mag_angle = np.zeros([self.mag_data.shape[0], 1])

        for i in range(self.acc_data.shape[0]):
            angle_all[i, 0] = np.arctan2(-self.acc_data[i, 1] ** 2.0, -self.acc_data[i, 2] ** 2.0)
            angle_all[i, 1] = np.arctan2(self.acc_data[i, 0],
                                         np.sqrt(self.acc_data[i, 1] ** 2.0 + self.acc_data[i, 2] ** 2.0))

            t_R = euler2mat(angle_all[i, 0], angle_all[i, 1], 0.0, 'sxyz')
            tmp_acc_data[i, :] = (t_R.dot(self.acc_data[i, :].transpose())).transpose()
            self.convert_mag_data[i, :] = (t_R.dot(self.mag_data[i, :].transpose())).transpose()

            self.angle[i, 0] = np.arcsin(
                self.convert_mag_data[i, 0] / np.linalg.norm(self.convert_mag_data[i, :2])) / np.pi
            self.direct_mag_angle[i, 0] = np.arctan2(self.convert_mag_data[i, 1], self.convert_mag_data[i, 0])

        self.zf = interpolate.interp1d(
            self.length_array[:, 0], self.convert_mag_data[:, 2] / self.convert_mag_data[:, 2].mean(), kind='linear')

        self.af = interpolate.interp1d(
            self.length_array[:, 0], self.angle[:, 0], kind='nearest'
        )

        if ifshow:

            plt.figure()
            plt.title('x-o-y angle mag')
            plt.plot(self.angle[:, 0], '-+')
            plt.grid()

            # plt.figure()
            # plt.title('angle')
            # for i in range(angle_all.shape[1]):
            #     plt.plot(angle_all[:, i], '-*', label=str(i))
            # plt.grid()
            # plt.legend()
            #
            # plt.figure()
            # plt.title('convert acc')
            # for i in range(tmp_acc_data.shape[1]):
            #     plt.plot(tmp_acc_data[:, i], '-+', label=str(i))
            # plt.grid()
            # plt.legend()
            #
            # plt.figure()
            # plt.title('acc')
            # for i in range(tmp_acc_data.shape[1]):
            #     plt.plot(self.acc_data[:, i], '-+', label=str(i))
            # plt.grid()
            # plt.legend()
            #
            plt.figure()
            plt.title('converted mag')
            for i in range(self.convert_mag_data.shape[1]):
                plt.plot(self.convert_mag_data[:, i], '-+', label=str(i))

            plt.grid()
            plt.legend()
            #
            # plt.figure()
            # plt.title('src mag')
            # for i in range(self.mag_data.shape[1]):
            #     plt.plot(self.mag_data[:, i], '-+', label=str(i))
            # plt.grid()
            # plt.legend()

    def GetZFFtDis(self, length, ifshow=True):

        tx = np.linspace(0.0, self.length_array[-1], num=self.length_array.shape[0] * 10)

        # self.mag_fft_list = list(self.length_array.shape[0])
        test_shape_fft = fft(np.linspace(0, length, int(length / 0.5)))
        self.mag_z_fft_feature = np.zeros([self.length_array.shape[0],
                                           len(test_shape_fft) ],
                                          dtype=np.complex)

        for i in range(0, self.length_array.shape[0]):

            if self.length_array[i] < length / 2.0 or \
                    self.length_array[i] > self.length_array[-1] - length / 2.0:
                continue
            else:
                the_x = np.linspace(self.length_array[i] - length / 2.0,
                                    self.length_array[i] + length / 2.0,
                                    int(length / 0.5))
                yyt = fft(self.zf(the_x))
                self.mag_z_fft_feature[i, :] = yyt

        self.tmp_fft_mat = self.ComputeDistanceFeatureSpace(self.mag_z_fft_feature)

        if ifshow:
            plt.figure()
            plt.title('dis z fft')

            plt.imshow(self.tmp_fft_mat)
            plt.colorbar()

            plt.figure()
            plt.title('gradient z')
            tmp_grandient = self.tmp_fft_mat[:, 1:] - self.tmp_fft_mat[:, :-1]
            ttmp_grandient = tmp_grandient[:, 1:] - tmp_grandient[:, :-1]

            plt.imshow((ttmp_grandient))
            plt.colorbar()

        return self.tmp_fft_mat

    def GetRelativeAttDis(self, length, ifshow=True):
        tx = np.linspace(0.0, self.length_array[-1], num=self.length_array.shape[0] * 10)

        # self.mag_fft_list = list(self.length_array.shape[0])
        test_shape_fft = fft(np.linspace(0, length, int(length / 0.5)))
        self.mag_att_feature = np.zeros([self.length_array.shape[0],
                                         len(test_shape_fft)])

        for i in range(0, self.length_array.shape[0]):

            if self.length_array[i] < length / 2.0 or \
                    self.length_array[i] > self.length_array[-1] - length / 2.0:
                continue
            else:
                the_x = np.linspace(self.length_array[i] - length / 2.0,
                                    self.length_array[i] + length / 2.0,
                                    int(length / 0.5))
                yyt = (self.af(the_x))
                self.mag_att_feature[i, :] = np.arcsin(np.abs(np.sin(yyt - self.af(self.length_array[i])))) / yyt.std()

        # self.tmp_fft_mat = self.ComputeDistanceFeatureSpace(self.mag_att_fft_feature)
        self.tmp_fft_mat = np.zeros([self.mag_att_feature.shape[0],
                                     self.mag_att_feature.shape[0]])
        for i in range(self.tmp_fft_mat.shape[0]):
            for j in range(i, self.tmp_fft_mat.shape[1]):
                self.tmp_fft_mat[i, j] = np.linalg.norm(
                    np.arcsin(np.abs(np.sin(
                        (self.mag_att_feature[i, :] - self.mag_att_feature[j, :])
                    )))
                )
                self.tmp_fft_mat[j, i] = self.tmp_fft_mat[i, j]

        if ifshow:
            plt.figure()
            plt.title('dis att fft')

            plt.imshow(self.tmp_fft_mat)
            plt.colorbar()

            plt.figure()
            plt.title('gradient z')
            tmp_grandient = self.tmp_fft_mat[:, 1:] - self.tmp_fft_mat[:, :-1]
            ttmp_grandient = tmp_grandient[:, 1:] - tmp_grandient[:, :-1]

            plt.imshow((ttmp_grandient))
            plt.colorbar()

        return self.tmp_fft_mat

    def ConvertMagAttitude(self):
        print('empty ...')
