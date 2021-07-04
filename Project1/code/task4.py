import os
from load_data import load_data
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import data_utils




if __name__=='__main__':
    
    P_cam20 = data_utils.calib_cam2cam('../data/problem_4/calib_cam_to_cam.txt', '02')
    #P_cam20 = np.concatenate([P_cam20,np.zeros((3,1))],axis=1)
    R,T = data_utils.calib_velo2cam('../data/problem_4/calib_velo_to_cam.txt')
    T_cam0_velo = np.concatenate([R,T],axis=1)
    T_cam0_velo = np.concatenate([T_cam0_velo,np.asarray([[0.,0.,0.,1]])],axis=0)
    R,T = data_utils.calib_velo2cam('../data/problem_4/calib_imu_to_velo.txt')
    T_velo_imu = np.concatenate([R,T],axis=1)
    T_imu_velo = np.concatenate([np.linalg.inv(R),-np.linalg.inv(R)@T],axis=1)
    T_velo_imu = np.concatenate([T_velo_imu,np.asarray([[0.,0.,0.,1]])],axis=0)
    T_imu_velo = np.concatenate([T_imu_velo,np.asarray([[0.,0.,0.,1]])],axis=0)

    # for each frame
    for ind in range(37,38):

        # get time stamp
        time_stamp_start = data_utils.compute_timestamps('../data/problem_4/velodyne_points/timestamps_start.txt', ind)
        time_stamp_end = data_utils.compute_timestamps('../data/problem_4/velodyne_points/timestamps_end.txt', ind)
        time_stamp = data_utils.compute_timestamps('../data/problem_4/image_02/timestamps.txt', ind)

        # load pcd data
        pcd = data_utils.load_from_bin("../data/problem_4/velodyne_points/data/"+str(ind).zfill(10)+".bin")

        # load image
        image = cv2.imread("../data/problem_4/image_02/data/"+str(ind).zfill(10)+".png")
        H = image.shape[0]
        W = image.shape[1]
        # print(W)
        # print(H)

        # compute depth
        pcd_d = np.linalg.norm(pcd,axis=1)

        # compute angle of the points cloud
        angle = np.arctan2(pcd[:,1],-pcd[:,0]) + math.pi #angle from x axis
        # compute start angle
        start_angle = (time_stamp_start-time_stamp)/(time_stamp_end-time_stamp_start)*2*math.pi
        #print(start_angle)
        

        # get time from time stamp of the point cloud
        t = (time_stamp_end - time_stamp_start) * (angle) /2./math.pi 
        mask = (angle > 2 * math.pi + start_angle).tolist()
        t[mask] = (time_stamp_end - time_stamp_start) * (angle[mask]-2*math.pi) /2./math.pi


        # get velocities of the frame
        speed_flu = data_utils.load_oxts_velocity("../data/problem_4/oxts/data/"+str(ind).zfill(10)+".txt")
        angular_rate_f, angular_rate_l, angular_rate_u = data_utils.load_oxts_angular_rate("../data/problem_4/oxts/data/"+str(ind).zfill(10)+".txt")


        # get the relative transformation using speed
        #rotation radius:
        r = np.sqrt(speed_flu[0]**2+speed_flu[1]**2)/angular_rate_u
        theta = angular_rate_u * t
        alpha = math.atan2(speed_flu[1], speed_flu[0])
        #alpha=0

        p = t[:,np.newaxis] @ speed_flu[:,np.newaxis].T
        # px = (np.sin(theta) * r).reshape((-1,1))
        # py = ((1 - np.cos(theta)) * r).reshape((-1,1))
        # pz = speed_flu[2] * t.reshape((-1,1))
        # p = np.concatenate([px,py,pz],axis=1)
        yaw = angular_rate_u * t

        uv_cam2 = []
        o_uv_cam2 = []
        depths = []
        o_depths = []
        for i in range(yaw.shape[0]):
            a = yaw[i]
            # transformation from position to the time stamp frame
            T_f_p = np.asarray([[math.cos(a),-math.sin(a),0.,p[i,0]],
                            [math.sin(a),math.cos(a),0.,p[i,1]],
                            [0.,0.,1.,p[i,2]],
                            [0.,0.,0.,1.]])
            # T_alpha = np.asarray([[math.cos(alpha),-math.sin(alpha),0.,0.],
            #                 [math.sin(alpha),math.cos(alpha),0.,0.],
            #                 [0.,0.,1.,0.],
            #                 [0.,0.,0.,1.]])
            T_alpha = np.eye(4)

            # project the point cloud in camera

            n_p = np.ones((4,1))
            n_p[0:3,:] = pcd[i,:].reshape((-1,1))
            n_uv = P_cam20 @ T_cam0_velo @ T_velo_imu @ T_alpha @ T_f_p @ T_imu_velo @ n_p
            uv =(n_uv[0:2]/n_uv[2]).astype(int)

            o_n_uv = P_cam20 @ T_cam0_velo @ n_p
            o_uv =(o_n_uv[0:2]/o_n_uv[2]).astype(int)


            if pcd[i,0]>0 and uv[0]>=0 and uv[0]<W and uv[1]>=0 and uv[1]<H:
                uv_cam2.append(uv)
                depths.append(pcd_d[i])

            if pcd[i,0]>0 and o_uv[0]>=0 and o_uv[0]<W and o_uv[1]>=0 and o_uv[1]<H:
                o_uv_cam2.append(o_uv)
                o_depths.append(pcd_d[i])

        # assign color
        uv_cam2 = np.asarray(uv_cam2).T
        uv_cam2 = uv_cam2[0,:,:]
        o_uv_cam2 = np.asarray(o_uv_cam2).T
        o_uv_cam2 = o_uv_cam2[0,:,:]
        colors = data_utils.depth_color(np.asarray(depths), max_d=70)
        final_image = data_utils.print_projection_plt(uv_cam2, colors, image)
        o_colors = data_utils.depth_color(np.asarray(o_depths), max_d=70)
        o_final_image = data_utils.print_projection_plt(o_uv_cam2, o_colors, image)

        if ind==37:
            cv2.imwrite('../result/task4_undistorted.jpg', final_image)
            cv2.imwrite('../result/task4_distorted.jpg', o_final_image)

        
        
    

   