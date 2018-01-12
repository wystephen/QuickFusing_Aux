# -*- coding:utf-8 -*-
# Created by steve @ 18-1-12 上午9:36
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
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class TraceCompare(object):



    def __init__(self,src_trace,target_trace):
        self.src_trace = src_trace
        self.target_trace = target_trace

        res = minimize(self.tError,x0=np.zeros([3]),
                       method='Powell')
        # print(res)
        t  = res.x
        plt.figure()
        plt.plot(self.src_trace[:,0],self.src_trace[:,1],label='src')
        plt.plot(self.target_trace[:,0],self.target_trace[:,1],label='target')
        t_target = self.rotate_2d(self.target_trace[:,:2]+t[:2],t[2])
        plt.plot(t_target[:,0],t_target[:,1],label='modified target')
        plt.grid()
        plt.legend()
        self.error= self.tError(t)





    def tError(self,t):

        '''

        :param t:
        :return:
        '''
        target_trace = self.rotate_2d(self.target_trace[:,:2]+t[:2],t[2])

        src_trace = self.src_trace[:,:2]

        return np.mean(np.linalg.norm(src_trace-target_trace,axis=1))



    def rotate_2d(self, trace, angle):
        '''
        rotate serious of points according to given angle(in radio)
        :param trace:
        :param angle:
        :return:
        '''
        tmp_trace = np.zeros_like(trace[:, :2])

        tmp_trace[:, 0] = trace[:, 0] * np.cos(angle) - \
                          trace[:, 1] * np.sin(angle)
        tmp_trace[:, 1] = trace[:, 1] * np.cos(angle) + \
                          trace[:, 0] * np.sin(angle)
        return tmp_trace.copy()

