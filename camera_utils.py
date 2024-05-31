import numpy as np

d405_K = np.load("handeye/intrinsics_d405.npy")
d405_T_C_EEF = np.load("handeye/T_C_EEF_wrist_l.npy")
d415_T_WC = np.load("handeye/T_WC_head.npy")

def convert_from_uvd(K, u, v, d):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_over_z = (u - cx) / fx
    y_over_z = (v - cy) / fy
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
    
    return x, y, z