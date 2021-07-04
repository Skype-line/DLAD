from load_data import load_data
import numpy as np
import cv2
import os

"""Exercise 1 Task 3 for DLAD course"""

def main():
	# load data
	data_path = os.path.join("../data", "data.p")
	data = load_data(data_path)
	pcd = data["velodyne"]		# point cloud (119454, 4)  [x, y, z, reflectance intensity]
	P20 = data["P_rect_20"]		# projection from 3D points in Cam 0 to 2D image points in Cam 2
	T0v = data["T_cam0_velo"]	# transformation from velo to Cam 0
	image2 = data["image_2"]	# RGB image of Cam 2 (376, 1241, 3)

	# filter out the point behind the camera
	filtered_pcd = pcd[pcd[:,0]>0.27][:,0:3]
	homo_pcd = np.concatenate((filtered_pcd, np.ones((len(filtered_pcd),1))),axis=1)
	# project 3D points in Cam 0 to 2D image points in Cam 2
	homo_p2d = P20 @ T0v @ (homo_pcd.T)
	p2d = ((homo_p2d/homo_p2d[-1,:])[0:2,:]).T.astype(int)
	# filter out image points beyond the scope of image and their corresponding 3D points
	mask1 = np.logical_and(p2d[:,0]>0, p2d[:,1]>0)
	mask2 = np.logical_and(p2d[:,0]<image2.shape[1], p2d[:,1]<image2.shape[0])
	mask = np.logical_and(mask1,mask2)
	filtered_p2d = p2d[mask]
	filtered_p3d = filtered_pcd[mask]
	# compute the vertical angle
	angles = np.arctan2(filtered_p3d[:,2],np.sqrt(np.power(filtered_p3d[:,0],2)+np.power(filtered_p3d[:,1],2)))
	# divide true FOV into 64 equal ranges
	angle_res = (np.max(angles)-np.min(angles))/64
	# compute Laser ID for each point, {1,...,64}
	laser_ID = ((angles-np.min(angles))/angle_res).astype(int)

	# manually implement the plot func instead of using given functions
	# use four alternating colors to indicate the identified IDs
	laser_colorID = np.mod(laser_ID,4)
	laser_color_map = {0:[0,0,255],1:[255,0,0],2:[0,255,0],3:[255,0,255]}
	laser_colors = np.array([laser_color_map[laser_colorID[i]] for i in range(len(laser_colorID))])
	for idx, point in enumerate(filtered_p2d):
		color = laser_colors[idx,:]
		color = (int(color[0]), int(color[1]), int(color[2])) 
		cv2.circle(image2,tuple(point),1,tuple(color))
	cv2.imwrite("../result/task3.png",image2)

if __name__ == '__main__':
	main()