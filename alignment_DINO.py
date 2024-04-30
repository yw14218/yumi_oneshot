import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
from torchvision.datasets import ImageFolder
import warnings 
import glob
import time
warnings.filterwarnings("ignore")


from PIL import Image
import torchvision.transforms as T
# import hubconf
#Install this DINO repo to extract correspondences: https://github.com/ShirAmir/dino-vit-features
from DinoViT.correspondences import find_correspondences, draw_correspondences


#Here are the functions you need to create based on your setup.
def camera_get_rgbd():
    #Outputs a tuple (rgb, depth) taken from a wrist camera.
    #The two observations should have the same dimension.
    raise NotImplementedError
    
def add_depth(points, depth):
    """
    Inputs: points: list of [x,y] pixel coordinates, 
    	    depth (H,W,1) observations from camera.
    Outputs: point_with_depth: list of [x,y,z] coordinates.
    
    Adds the depth value/channel to the list of pixels by
    extracting the corresponding depth value from depth.
    """
    
    raise NotImplementedError 
    
def convert_pixels_to_meters(t):
	"""
	Inputs: t : (x,y,z) translation of the end-effector in pixel-space/frame
	Outputs: t_meters : (x,y,z) translation of the end-effector in world frame
	
	Requires camera calibration to go from a pixel distance to world distance
	"""
	raise NotImplementedError 
	
	
def robot_move(t_meters,R):
	"""
	Inputs: t_meters: (x,y,z) translation in end-effector frame
			R: (3x3) array - rotation matrix in end-effector frame
	
	Moves and rotates the robot according to the input translation and rotation.
	"""
	raise NotImplementedError
	

def find_transformation(X, Y):
    #Find transformation given two sets of correspondences between 3D points
    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t

def compute_error(points1, points2):
	return np.linalg.norm(np.array(points1) - np.array(points2))

#Hyperparameters for DINO correspondences extraction
num_pairs = 8 #@param
load_size = 224 #@param
layer = 9 #@param
facet = 'key' #@param
bin=True #@param
thresh=0.05 #@param
model_type='dino_vitb8' #@param
stride=4 #@param
if __name__ == '__main__':
    
 #Get rgbd from wrist camera.
 rgb_bn, depth_bn = camera_get_rgbd()
 Error = 100000
 ERR_THRESHOLD = 50 #A generic error between the two sets of points
 while error > ERR_THRESHOLD:
   rgb_live, depth_live = camera_get_rgbd()
   with torch.no_grad():
                points1, points2, image1_pil, image2_pil = find_correspondences(rgb_live, rgb_bn, num_pairs, load_size, layer,
                                                                               facet, bin, thresh, model_type, stride)
   #Given the pixel coordinates of the correspondences, add the depth channel
   points1 = add_depth(points1, depth_bn)
   points2 = add_depth(points2, depth_live)
   R, t = find_transformation(points1, points2)
   #A function to convert pixel distance into meters based on calibration of camera.
   t_meters = convert_pixels_to_meters(t)
   #Move robot
   robot.move(t_meters,R)
   error = compute_error(points1, points2)
