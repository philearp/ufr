import numpy as np
import matplotlib.pyplot as plt

def read_calibration_data_file(filename):
    raw_data = np.loadtxt(filename, delimiter=',')
    
    data = {}
    data['theta_x'] # roll, [degrees]
    data['theta_y'] # pitch, [degrees]
    data['qpd_x'] = data[:,2]
    data['qpd_y'] = data[:,3]
    data['qpd_sum'] = data[:,4]

    return data