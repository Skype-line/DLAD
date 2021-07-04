# Deep Learning for Autonomous Driving
# Material for Problem 2 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import os
from load_data import load_data
import math

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

    def update(self, points, sem_labels, color_map):
        '''
        :param points: point cloud data
                        shape (N, 3)          
        Task 2: Change this function such that each point
        is colored depending on its semantic label
        '''
        # define colors
        N = points.shape[0]
        colors = np.zeros((N,3))
        #get keys
        keys = list(color_map.keys())
        for key in keys:
            colors[(sem_labels==key).reshape((-1,)).tolist(),:] = color_map[key]
        
        colors = colors/255.
        RGB = np.zeros((N,3))
        print(RGB.shape)
        RGB[:,0] = colors[:,-1]
        RGB[:,1] = colors[:,1]
        RGB[:,-1] = colors[:,0]
        self.sem_vis.set_data(points, size=3, face_color = RGB)
    
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
        self.obj_vis.set_data(corners.reshape(-1,3),
                              connect=connect,
                              width=2,
                              color=[0,1,0,1])

if __name__ == '__main__':
    data = load_data('data/data.p') # Change to data.p for your final submission 
    visualizer = Visualizer()
    visualizer.update(data['velodyne'][:,:3], data['sem_label'], data['color_map'])
    '''
    Task 2: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''
    
    objects = data['objects']
    corners = np.zeros((len(objects),8,3))
    bboxs = [objects[i][8:15:1] for i in range(len(objects))]
    i=0
    for bbox in bboxs:
        #get point grid
        p_grid = [[-1,-1,-1],
                  [-1,1,-1],
                  [1,1,-1],
                  [1,-1,-1],
                  [-1,-1,1],
                  [-1,1,1],
                  [1,1,1],
                  [1,-1,1]]
        p_grid = np.asarray(p_grid).astype(float)
        whl = np.asarray([bbox[2],bbox[0], bbox[1]]).reshape((1,-1))
        #get point grid in obj coordinate, each line a point
        p_grid_obj = np.multiply(p_grid, whl) * 0.5
        n_p_grid_obj = np.concatenate([p_grid_obj,np.ones((8,1))],axis=1)

        #transform to velodyne
        t = bbox[-1]
        T_cam0_obj = np.asarray([[math.cos(t),0.,math.sin(t),bbox[3]],
                                 [0.,1.,0.,bbox[4]-0.5*bbox[0]],
                                 [-math.sin(t),0.,math.cos(t),bbox[5]],
                                 [0.,0.,0.,1.]])
        T_cam0_velo = data['T_cam0_velo']
        n_p_grid_velo = (np.linalg.inv(T_cam0_velo) @ T_cam0_obj @ n_p_grid_obj.T).T
        corners[i,:,:] = n_p_grid_velo[:,0:3]

        i+=1

    
    visualizer.update_boxes(corners)


    vispy.app.run()




