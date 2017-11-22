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

from scipy  import  interpolate
from scipy.fftpack import fft,ifft

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
                                      [:,0],self.mag_norm,kind='cubic')

        plt.figure()
        plt.title('inter')
        tx = np.linspace(0.0,self.length_array[-1],num=self.length_array.shape[0]*10)












        plt.plot(tx,self.f(tx),'r+',label='interp')
        plt.plot(self.length_array,self.mag_norm,'b*',label='source mag norm')
        plt.legend()
        plt.grid()

