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

import matplotlib.pyplot as plt
import seaborn as sbn


def lose_func(v):
    if v < 1.0:
        return 0.0
    elif v < 2.0:
        return v - 1.0
    else:
        return (v - 1.0) ** 0.3


def kernel_fuc(v):
    return min(2 * 1.0 / (1.0 + v * v), 1.0) * v


if __name__ == '__main__':
    sbn.set("paper", "whitegrid")

    x = np.linspace(0.0, 10.0, 1000000)
    # x =
    y1 = np.vectorize(kernel_fuc)(x)
    y2 = np.vectorize(lose_func)(x)

    legend_fond_size = 10
    axis_a = [0.0, 10.0, -1.0, 2.0]
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title(r'(a)')
    plt.plot(x, y1, label=r'$e_{loopKernel}$')
    plt.plot(x[1:], (y1[1:] - y1[:-1]) / (x[1:] - x[:-1]),
             label=r'${\partial{e_{loopKernel}}}/{\partial{e_{loss}}}$')

    plt.legend(fontsize=legend_fond_size)
    plt.axis(axis_a)

    plt.subplot(1, 2, 2)
    plt.title(r'(b)')
    plt.plot(x, y2, label=r'$e_{lossloop}$')
    plt.plot(x[1:], (y2[1:] - y2[:-1]) / (x[1:] - x[:-1]),
             label=r'${\partial{e_{lossloop}}}/{\partial{e_{loss}}}$')

    plt.legend(fontsize=legend_fond_size)
    plt.axis(axis_a)

    plt.savefig('test.png', dpi=1000)
    plt.show()
