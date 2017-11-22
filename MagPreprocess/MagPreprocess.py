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



    def Step2Length(self,length):

