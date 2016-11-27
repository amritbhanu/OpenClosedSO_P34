__author__ = 'amrit'

import matplotlib.pyplot as plt
import os, pickle
import operator
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors

if __name__ == '__main__':

    F_final1={}
    path = '../dump/'
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            a = os.path.join(root, name)
            with open(a, 'rb') as handle:
                F_final = pickle.load(handle)
                for i,k in F_final.iteritems():
                    print(i)
                    for m,n in k.iteritems():
                        print(m)
                        print(n)