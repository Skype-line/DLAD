import os
from load_data import load_data
import numpy as np
import cv2




if __name__=='__main__':
    data_path = os.path.join('../data','data.p')
    data = load_data(data_path)

    #read pointcloud data
    pcd = data['velodyne']

    #get range of data
    x_max = np.max(pcd[:,0])
    x_min = np.min([pcd[:,0]])
    y_max = np.max(pcd[:,1])
    y_min = np.min(pcd[:,1])

    H = int((x_max - x_min)/0.2)
    W = int((y_max - y_min)/0.2)

    #get u, v:
    U = ((pcd[:,0]-x_min)/0.2).astype(int)
    V = ((pcd[:,1]-y_min)/0.2).astype(int)

    U_ref = []
    V_ref = []
    value_ref = []


    #sort the array
    UV = np.concatenate([U.reshape((-1,1)),V.reshape((-1,1))],axis=1)
    pixel_index = UV[:,0]*W + UV[:,1]
    sort_index = pixel_index.argsort()
    UV = UV[sort_index,:]
    value = pcd[sort_index,3]
    #remove duplicate
    i=0
    while i<UV.shape[0]-1:
        j = i+1
        max_ref = value[i]
        while UV[i,0]==UV[j,0] and UV[i,1]==UV[j,1] and j<UV.shape[0]-1:
            if value[j]>max_ref:
                max_ref = value[j]
            j += 1

        U_ref.append(UV[i,0])
        V_ref.append(UV[i,1])
        value_ref.append(max_ref)
        i = j

    #build image
    BEV_image = np.zeros((H+1,W+1))
    BEV_image[U_ref,V_ref] = value_ref

    BEV_image = cv2.rotate(BEV_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite('../result/task1.jpg',np.uint8(BEV_image*255))
        

    