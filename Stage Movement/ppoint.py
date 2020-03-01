import numpy as np
import matplotlib.pyplot as plt

# Function Definitions

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

def calc_rotation_matrix(theta_x_deg, theta_y_deg, theta_z_deg):
    R_x = rotation_x(np.deg2rad(theta_x_deg))

    R_y = rotation_y(np.deg2rad(theta_y_deg))

    R_z = rotation_z(np.deg2rad(theta_z_deg))

    R = np.matmul(R_y, R_z)
    R = np.matmul(R_x, R)
    return R

# main code starts here:

# Define offset from stage rotation centre (needle tip) (C) to focal point of laser (A_ref)
CA_ref = np.array([80, 0, 250]) # [mm]

# Define desired user displacements
ux = 0 # [mm]
uy = 0 # [mm]
uz = 0 # [mm]

# Define desired user rotation angles
theta_x_deg = 1 # [degrees]
theta_y_deg = 0 # [degrees]
theta_z_deg = 0 # [degrees]

# Vector from reference position on sample (Q) to desired position on sample (P)
QP = np.array([ux, uy, uz]) # [mm]

# 3x3 Rotation matrix due to angles:
#  theta_x_deg about x-axis
#  theta_y_deg about y-axis
#  theta_z_deg about z-axis
R = calc_rotation_matrix(theta_x_deg, theta_y_deg, theta_z_deg)

# Calculate required stage displacement to position laser focus point at desired position on sample
OC = QP + CA_ref - np.matmul(R, CA_ref)

print("Requires stage translation of:")
print("x = {0:.3f}mm".format(OC[0]))
print("y = {0:.3f}mm".format(OC[1]))
print("z = {0:.3f}mm".format(OC[2]))