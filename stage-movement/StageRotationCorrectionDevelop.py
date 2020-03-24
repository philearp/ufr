# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:37:53 2019

@author: pearp
"""
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt

## user inputs
theta_x_deg = 1
theta_y_deg = 1
theta_z_deg = 0
##

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

## Start of code

A = np.array([15, 5, 130]) # offset from stage rotation centre to desired rotation centre
print("Rotation centre offset is:")
print("x = " + str(A[0]) + "mm")
print("y = " + str(A[1]) + "mm")
print("z = " + str(A[2]) + "mm")

print("For an rotation angle of ")
print(str(theta_x_deg) + " degree about the x-axis,")
print(str(theta_y_deg) + " degree about the y-axis, and")
print(str(theta_z_deg) + " degree about the z-axis:")

# Calculate elements of rotation matrices
R_x = rotation_x(np.deg2rad(theta_x_deg))

R_y = rotation_y(np.deg2rad(theta_y_deg))

R_z = rotation_z(np.deg2rad(theta_z_deg))

R = np.matmul(R_y, R_z)
R = np.matmul(R_x, R)

# Apply rotation matrix R to point A
B = np.matmul(R, A) # location of desired rotation centre after stage rotation

T_ab = B - A 
T_ba = -T_ab # desired post-rotation translation to move rotation centre back to axis

print("Requires stage translation of:")
print("x = {0:.3f}mm".format(T_ba[0]))
print("y = {0:.3f}mm".format(T_ba[1]))
print("z = {0:.3f}mm".format(T_ba[2]))