# Deep Learning for Autonomous Driving
# Material for Problem 2 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import os
from load_data import load_data

class Visualizer():
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)
        
        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray([[0,1],[0,3],[0,4],
                                   [2,1],[2,3],[2,6],
                                   [5,1],[5,4],[5,6],
                                   [7,3],[7,4],[7,6]])

    def update(self, points, semantic_labels, color_map):
        '''
        :param points: point cloud data
                        shape (N, 3)  
        :colors: color for each point ()        
        Task 2.3: Change this function such that each point
        is colored depending on its semantic label
        '''
        p_colors_BGR = np.array([color_map[semantic_labels[i][0]] for i in range(len(semantic_labels))])/255
        p_colors_RGB = np.concatenate([p_colors_BGR[:,2:3],p_colors_BGR[:,1:2],p_colors_BGR[:,0:1]],axis=1)
        self.sem_vis.set_data(points, size=3, edge_width=0.2, edge_color=p_colors_RGB, face_color=p_colors_RGB)
    
    def update_boxes(self, corners):
        '''
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        If you plan to use a different order, you can
        change self.connect accordinly.
        '''
        for i in range(corners.shape[0]):
            connect = np.concatenate((connect, self.connect+8*i), axis=0) \
                      if i>0 else self.connect
        print(connect)
        self.obj_vis.set_data(corners.reshape(-1,3),
                              connect=connect,
                              width=2,
                              color=[0,1,0,1])

if __name__ == '__main__':
    data = load_data('../data/data.p') # Change to data.p for your final submission 
    visualizer = Visualizer()
    visualizer.update(data['velodyne'][:,:3],data["sem_label"],data["color_map"])
    '''
    Task 2.3: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''
    corners = []
    objects = data["objects"]
    T0v = data["T_cam0_velo"]
    # compute transformation from Cam 0 to velo
    R=T0v[0:3,0:3]
    t=T0v[0:3,3:4]
    Tv0 = np.block([[R.T,-R.T@t],
                   [0, 0, 0, 1]])
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
      corners_rela = np.array([[L/2,  L/2,  L/2, L/2, -L/2, -L/2, -L/2, -L/2],
                         [0,    -H,   -H,  0,     0,  -H  , -H  ,   0    ],
                         [-W/2, -W/2, W/2, W/2, -W/2, -W/2, W/2 , W/2],
                         [1,    1,    1,   1,     1,    1,   1,    1     ]])
      # get the 3D position of 8 points in Cam 0
      corners3D = rotation @ corners_rela + translation
      # get the 3D position of 8 points in velo
      corners3D = Tv0 @ corners3D
      corners.append(corners3D[0:3,:].T)
    visualizer.update_boxes(np.array(corners))
    vispy.app.run()




