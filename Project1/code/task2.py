from load_data import load_data
import numpy as np
import cv2
import os

"""Exercise 1 Task 2 for DLAD course"""

def draw_bbx(image, p, color, thickness):
	cv2.line(image,tuple(p[0,:]),tuple(p[1,:]),color,thickness)
	cv2.line(image,tuple(p[1,:]),tuple(p[2,:]),color,thickness)
	cv2.line(image,tuple(p[2,:]),tuple(p[3,:]),color,thickness)
	cv2.line(image,tuple(p[3,:]),tuple(p[0,:]),color,thickness)
	cv2.line(image,tuple(p[4,:]),tuple(p[5,:]),color,thickness)
	cv2.line(image,tuple(p[5,:]),tuple(p[6,:]),color,thickness)
	cv2.line(image,tuple(p[6,:]),tuple(p[7,:]),color,thickness)
	cv2.line(image,tuple(p[7,:]),tuple(p[4,:]),color,thickness)
	cv2.line(image,tuple(p[0,:]),tuple(p[4,:]),color,thickness)
	cv2.line(image,tuple(p[1,:]),tuple(p[5,:]),color,thickness)
	cv2.line(image,tuple(p[2,:]),tuple(p[6,:]),color,thickness)
	cv2.line(image,tuple(p[4,:]),tuple(p[7,:]),color,thickness)

def main():
	# load data
	data_path = os.path.join("../data", "data.p")
	data = load_data(data_path)
	pcd = data["velodyne"]			# point cloud (119454, 4)  [x, y, z, reflectance intensity]
	label = data["sem_label"]		# semantic label for each point (119454, 1) 
	P20 = data["P_rect_20"]			# projection from 3D points in Cam 0 to 2D image points in Cam 2
	T0v = data["T_cam0_velo"]		# transformation from velo to Cam 0
	image2 = data["image_2"]		# RGB image of Cam 2 (376, 1241, 3)
	Color_map = data["color_map"]	# map from semantic label to RGB color for visualization (34 labels)
	objects = data["objects"]		# list of object info (3 dims: H,W,L, 3 locs in Cam 0, 1 rotation_angle_y)

	"""Task 2.1: Visualization of 3D Semantic Segmentation on 2D image"""
	pcd_x = pcd[:,0]
	# filter out the point behind the camera
	filtered_pcd = pcd[pcd_x>0.27][:,0:3]
	label_pcd = label[pcd_x>0.27]
	homo_pcd = np.concatenate((filtered_pcd, np.ones((len(filtered_pcd),1))),axis=1)
	# project 3D points in Cam 0 to 2D image points in Cam 2
	homo_p2d = P20 @ T0v @ (homo_pcd.T)
	p2d = ((homo_p2d/homo_p2d[-1,:])[0:2,:]).T.astype(int)
	# filter out image points beyond the scope of image
	mask1 = np.logical_and(p2d[:,0]>0, p2d[:,1]>0)
	mask2 = np.logical_and(p2d[:,0]<image2.shape[1], p2d[:,1]<image2.shape[0])
	mask = np.logical_and(mask1,mask2)
	filtered_p2d = p2d[mask]
	# get the color for each remaining point
	p_colors = np.array([Color_map[label_pcd[i][0]] for i in range(len(label_pcd))])
	p_colors = p_colors[mask]
	# plot the points
	for idx, point in enumerate(filtered_p2d):
		color = p_colors[idx,:]
		color = (int(color[0]), int(color[1]), int(color[2])) 
		cv2.circle(image2,tuple(point),1,tuple(color))
	cv2.imwrite("../result/task2.1.png",image2)

	"""Task 2.2: Visualization of 3D Detection on 2D image """
	for obj in objects:
		# extract height, weight, length of the bounding box, location of its bottom center in Cam 0 and rotation angle round y axis
		H, W, L, xc, yc, zc, rotation_y = obj[-8:-1]
		# compute rotation and translation
		rotation = np.array([[np.cos(rotation_y),0,np.sin(rotation_y),0],
							[0,1,0,0],
							[-np.sin(rotation_y),0,np.cos(rotation_y),0],
							[0,0,0,1]])
		translation = np.array([[xc],[yc],[zc],[0]])
		# compute the position of eight corners relative to the bottom center
		corners = np.array([[L/2,  L/2,  L/2, L/2, -L/2, -L/2, -L/2, -L/2],
							[0,    -H,   -H,  0,     0,  -H  , -H  ,   0 ],
							[-W/2, -W/2, W/2, W/2, -W/2, -W/2, W/2 , W/2],
							[1,    1,    1,   1,     1,    1,   1,    1 ]])
		# get the 3D position of 8 points in Cam 0
		corners3D = rotation @ corners + translation
		# project 3D points in Cam 0 to 2D image points in Cam 2
		homo_corners = P20 @ corners3D
		corners2D = ((homo_corners/homo_corners[-1,:])[0:2,:]).T.astype(int)
		# draw bounding box
		draw_bbx(image2,corners2D,(0,255,0),2)
	cv2.imwrite("../result/task2.2.png",image2)

	"""Task 2.3: Visualization of 3D Semantic Segmentation and Detection in 3D space
	 run 3dvis.py!
	"""
		
if __name__ == '__main__':
	main()
