import numpy as np
import matplotlib.pyplot as plt

# Function Definitions

def c(theta):
    # theta [radians]
    return np.cos(theta)

def s(theta):
    # theta [radians]
    return np.sin(theta)

def rotation_x(theta):
    # theta [radians]
    R = np.array([[1, 0, 0], [0, c(theta), -s(theta)], [0, s(theta), c(theta)]])
    return R

def rotation_y(theta):
    # theta [radians]
    R = np.array([[c(theta), 0, s(theta)], [0, 1, 0], [-s(theta), 0, c(theta)]])
    return R

def rotation_z(theta):
    # theta [radians]
    R = np.array([[c(theta), -s(theta), 0], [s(theta), c(theta), 0], [0, 0, 1]])
    return R

def calc_rotation_matrix(theta_x_deg, theta_y_deg, theta_z_deg):
    R_x = rotation_x(np.deg2rad(theta_x_deg))

    R_y = rotation_y(np.deg2rad(theta_y_deg))

    R_z = rotation_z(np.deg2rad(theta_z_deg))

    R = np.matmul(R_y, R_z)
    R = np.matmul(R_x, R)
    return R

def main_ppoint_correction(CA_ref, QP, thetas_deg):
    # inputs:
    #   CA_ref = offset from stage rotation centre (needle tip) (C) to focal point of laser (A_ref) [mm]
    #   QP = Vector from reference position on sample (Q) to desired position on sample (P) [mm]
    #   thetas_deg = User tilt angles [degrees]

    # unpack thetas
    theta_x_deg = thetas_deg[0]
    theta_y_deg = thetas_deg[1]
    theta_z_deg = thetas_deg[2]

    # 3x3 Rotation matrix due to angles:
    #  theta_x_deg about x-axis
    #  theta_y_deg about y-axis
    #  theta_z_deg about z-axis
    R = calc_rotation_matrix(theta_x_deg, theta_y_deg, theta_z_deg)

    # Calculate required stage displacement to position laser focus point at desired position on sample
    OC = QP + CA_ref - np.matmul(R, CA_ref)

    return OC

# CODE BEGINS HERE

# User inputs:
# Define offset from stage rotation centre (needle tip) to focal point of laser
optics_offset = np.array([80, 0, 240]) # [mm]

# Define desired user displacements from reference position on sample
user_translation = np.array([0, 0, 0]) # [mm]

# Define desired user rotation angles
user_tilt = np.array([1, 0, 0]) #[degrees]

# Calculation of required stage translation:
print("P-Point translation of:")
print("({0:.2f}, {1:.2f}, {2:.2f}) mm.".format(user_translation[0], user_translation[1], user_translation[2]))
print("({0:.2f}, {1:.2f}, {2:.2f}) degrees,".format(user_tilt[0], user_tilt[1], user_tilt[2]))
print("for an optical set-up P-Point offset of ({0:.0f}, {1:.0f}, {2:.0f}) mm".format(optics_offset[0], optics_offset[1], optics_offset[2]))

required_stage_translation = main_ppoint_correction(optics_offset, user_translation, user_tilt)

print("r5equires stage translation of x = {0:.3f}mm, y = {1:.3f}mm, z = {2:.3f}mm".format(required_stage_translation[0], required_stage_translation[1], required_stage_translation[2]))


