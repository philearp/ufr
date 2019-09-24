# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:37:53 2019

@author: pearp
"""
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt

#def rotation_array(theta, axis):
#    l = axis[0]
#    m = axis[1]
#    n = axis[2]
#
#    R = np.array([[l*l*(1-np.cos(theta)) + np.cos(theta),    m*l*(1-np.cos(theta)) - n*np.sin(theta), n*l*(1-np.cos(theta)) + m*np.sin(theta)],
#                  [l*m*(1-np.cos(theta)) + n*np.cos(theta),  m*m*(1-np.cos(theta)) + np.cos(theta),   n*m*(1-np.cos(theta)) - l*np.sin(theta)],
#                  [l*n*(1-np.cos(theta)) - m*np.cos(theta),  m*n*(1-np.cos(theta)) + l*np.sin(theta), n*n*(1-np.cos(theta)) + np.cos(theta)]])
#    return R
def c(theta):
    return np.cos(theta)

def s(theta):
    return np.sin(theta)

def rotation_x(theta):
    R = np.array([[1, 0, 0], [0, c(theta), -s(theta)], [0, s(theta), c(theta)]])
    return R

def rotation_y(theta):
    R = np.array([[c(theta), 0, s(theta)], [0, 1, 0], [-s(theta), 0, c(theta)]])
    return R

def rotation_z(theta):
    R = np.array([[c(theta), -s(theta), 0], [s(theta), c(theta), 0], [0, 0, 1]])
    return R

A = np.array([15, 5, 130]) # offset from stage rotation centre to desired rotation centre

theta = np.deg2rad(1)

R = rotation_z(theta)

B = np.matmul(R, A) # location of desired rotation centre after stage rotation

T_ab = B - A 
T_ba = -T_ab # desired post-rotation translation to move rotation centre back to axis