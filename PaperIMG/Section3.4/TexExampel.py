# -*- coding:utf-8 -*-
# Created by steve @ 17-12-22 下午3:50
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
# -*- coding:utf-8 -*-
# Created by steve @ 17-12-22 下午3:27
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
# from __future__ import unicode_literals
import numpy as np
import scipy as sp


import matplotlib
# matplotlib.use('Ps')
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True

import matplotlib.pyplot as plt
import seaborn as sbn

def lose_func(v):
    if v<1.0:
        return 0.0
    elif v<2.0:
        return v-1.0
    else:
        return  (v-1.0)**0.3

if __name__ == '__main__':
    sbn.set("paper","whitegrid")
    # matplotlib.rc('text',usetex=True)

    x = np.linspace(0.0,10.0,1000000)
    # x =
    y2 = np.vectorize(lose_func)(x)


    plt.figure()
    plt.subplot(1,2,1)
    plt.title(r"\Tex e_{loopKernel}")
    plt.plot(x,x**2.0)

    plt.subplot(1,2,2)
    plt.title(r"e_{lossloop}")
    plt.plot(x,y2)
    plt.savefig('test.png')


    plt.show