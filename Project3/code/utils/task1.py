import numpy as np
from shapely.geometry import Polygon
import functools

def label2corners(label):
    '''
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame
    '''
    center = label[:, 0:3]
    bbox = label[:, 3:7]
    #get point grid (8,3)
    p_grid = [[1,-1,1],
                [1,-1,-1],
                [-1,-1,-1],
                [-1,-1,1],
                [1,0,1],
                [1,0,-1],
                [-1,0,-1],
                [-1,0,1]]
    p_grid = np.asarray(p_grid).astype(float)
    # lwh (N, 3)
    lhw = np.concatenate([bbox[:, 2, None], 2*bbox[:, 0, None], bbox[:, 1, None]], axis=1)
    #get point grid in obj coordinate, each line a point (N, 8, 3)
    p_grid_obj = np.multiply(p_grid, lhw.reshape((-1, 1, 3))) * 0.5

    #transform to cam0
    ry = bbox[:, 3:4] #(N, 1)

    # (N, 1) * (N, 8) + (N, 1) = (N, 8)
    X_cam0 = np.multiply(np.cos(ry), p_grid_obj[:,:,0]) + np.multiply(np.sin(ry), p_grid_obj[:,:,2]) + center[:,0:1]
    Y_cam0 = p_grid_obj[:,:,1] + center[:,1:2]
    Z_cam0 = np.multiply(-np.sin(ry), p_grid_obj[:,:,0]) + np.multiply(np.cos(ry), p_grid_obj[:,:,2]) + center[:,2:3]
    
    p_grid_cam0 = np.concatenate([X_cam0[:, :, None], Y_cam0[:, :, None], Z_cam0[:, :, None]], axis=2)
    
    return p_grid_cam0

    # N = label.shape[0]
    # corners = []
    # for i in range(N):
    #     x,y,z,h,w,l,ry = label[i,:]
    #     # compute rotation and translation
    #     rotation = np.array([[np.cos(ry),0,np.sin(ry)],
    #                         [0,1,0],
    #                         [-np.sin(ry),0,np.cos(ry)]])
    #     translation = np.array([[x],[y],[z]])
    #     # compute the position of eight corners relative to the bottom center
    #     corners_rel = np.array([[l/2,  l/2,  -l/2, -l/2, l/2,  l/2, -l/2, -l/2],
    #                             [ -h,  -h ,   -h ,  -h ,  0 ,   0 ,   0 ,   0 ],
    #                             [w/2, -w/2,  -w/2,  w/2, w/2, -w/2, -w/2,  w/2]])
    #     # get the 3D position of 8 points in Cam 0
    #     corners3D = rotation @ corners_rel + translation
    #     corners.append(corners3D.T)
    # return np.array(corners)

def cal_iou3d(corners_pred, corners_targ, vol_pred, vol_targ):
    '''
    input
        corners_pred (8x3)
        corners_targ (8x3)
        vol_pred     (1)
        vol_targ     (1)
    output
        iou_3d (1): iou of a 3D bounding box
    '''
    ymax1 = corners_pred[4,1]
    ymin1 = corners_pred[0,1]
    ymax2 = corners_targ[4,1]
    ymin2 = corners_targ[0,1]
    y_overlap = max(min(ymax1, ymax2) - max(ymin1, ymin2),0)
    pred_2D_box = Polygon([(corners_pred[0,0],corners_pred[0,2]),
                            (corners_pred[1,0],corners_pred[1,2]),
                            (corners_pred[2,0],corners_pred[2,2]),
                            (corners_pred[3,0],corners_pred[3,2])])
    targ_2D_box = Polygon([(corners_targ[0,0],corners_targ[0,2]),
                            (corners_targ[1,0],corners_targ[1,2]),
                            (corners_targ[2,0],corners_targ[2,2]),
                            (corners_targ[3,0],corners_targ[3,2])])
    inter_vol = pred_2D_box.intersection(targ_2D_box).area * y_overlap
    iou_3d = inter_vol / (vol_pred + vol_targ - inter_vol)
    return iou_3d

def get_iou(pred, target):
    '''
    Task 1
    input
        pred (N,7) 3D bounding box corners
        target (M,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    '''
    N = pred.shape[0]
    M = target.shape[0]
    iou = np.zeros([N,M])
    vol_pred = pred[:,3] * pred[:,4] * pred[:,5]       # (N)
    vol_targ = target[:,3] * target[:,4] * target[:,5] # (M)
    corners_pred = label2corners(pred)    # (N,8,3)
    corners_targ = label2corners(target)  # (M,8,3)
    for i in range(N):
        for j in range(M):
            iou[i,j] = cal_iou3d(corners_pred[i,:,:], corners_targ[j,:,:], vol_pred[i], vol_targ[j])
    return iou

def compute_recall(pred, target, threshold):
    '''
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene
    '''
    iou = get_iou(pred, target)
    iou_above_t = (iou > threshold).astype(int)
    
    # compute sum for each column to compute FN
    col_sum = np.sum(iou_above_t, axis=0)

    FN = np.sum(col_sum == 0)
    TP = np.sum(col_sum > 0)
    
    return float(TP / (TP + FN))
    
# """
# numpy implementation of 2d box intersection
# https://github.com/lilanxiao/Rotated_IoU/blob/c3dc561d3a2f727278181159f9cff2faac5c49f9/utiles.py
# author: lanxiao li
# 2020.8
# """
# EPSILON = 1e-8

# def box_intersection(corners1, corners2):
#     """find intersection points pf two boxes
#     Args:
#         corners1 (np.array): 4x2 coordinates of corners
#         corners2 (np.array): 4x2 coordinates of corners
#     Returns:
#         inters (4, 4, 2): (i, j, :) means intersection of i-th edge of box1 with j-th of box2
#         mask (4, 4) bool: (i, j) indicates if intersection exists 
#     """
#     assert corners1.shape == (4,2) 
#     assert corners2.shape == (4,2)
#     # build edges from corners
#     line1 = np.concatenate((corners1, corners1[[1,2,3,0],:]),axis=1) # (4, 4)
#     line2 = np.concatenate((corners2, corners2[[1,2,3,0],:]),axis=1) # (4, 4)
#     line1_ext = np.repeat(line1[:,np.newaxis,:],4,axis=1) # (4, 4) -> (4, 4, 4)
#     line2_ext = np.repeat(line2[np.newaxis,:,:],4,axis=0) # (4, 4) -> (4, 4, 4)
#     x1 = line1_ext[..., 0]
#     y1 = line1_ext[..., 1]
#     x2 = line1_ext[..., 2]
#     y2 = line1_ext[..., 3]
#     x3 = line2_ext[..., 0]
#     y3 = line2_ext[..., 1]
#     x4 = line2_ext[..., 2]
#     y4 = line2_ext[..., 3]
#     # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
#     num = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)     # (4, 4)
#     den_t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
#     t = den_t / (num + EPSILON)
#     t[num == .0] = -1.
#     mask_t = (t > 0) * (t < 1)                # intersection on line segment 1
#     den_u = (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)
#     u = -den_u / (num + EPSILON)
#     u[num == .0] = -1.
#     mask_u = (u > 0) * (u < 1)                # intersection on line segment 2
#     mask = mask_t * mask_u 
#     # t = den_t / (num + EPSILON)                 # overwrite with EPSILON. otherwise numerically unstable
#     intersections = np.stack([x1 + t*(x2-x1), y1 + t*(y2-y1)], axis=-1)
#     intersections = intersections * mask.astype(float)[:,:,np.newaxis]
#     return intersections, mask

# def box1_in_box2(corners1, corners2):
#     """check if corners of box1 lie in box2
#     Convention: if a corner is exactly on the edge of the other box, it's also a valid point
#     Args:
#         corners1 (np.array): 4x2 coordinates of corners
#         corners2 (np.array): 4x2 coordinates of corners
#     Returns:
#         c1_in_2 (4, ) Bool: i-th corner of box1 in box2
#     """
#     assert corners1.shape == (4,2) 
#     assert corners2.shape == (4,2)
#     a = corners2[0:1, :] # (1,2)
#     b = corners2[1:2, :] # (1,2)
#     d = corners2[3:4, :] # (1,2)
#     ab = b - a           # (1,2)
#     am = corners1 - a    # (4,2) 
#     ad = d - a           # (1,2)
#     p_ab = np.sum(ab * am, axis=-1)       # (4,)
#     norm_ab = np.sum(ab * ab, axis=-1)    # (1,)
#     p_ad = np.sum(ad * am, axis=-1)       # (4,)
#     norm_ad = np.sum(ad * ad, axis=-1)    # (1,)
#     # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
#     # also stable with different scale of bboxes
#     cond1 = (p_ab / norm_ab > - EPSILON) * (p_ab / norm_ab < 1 + EPSILON)   # (4,)
#     cond2 = (p_ad / norm_ad > - EPSILON) * (p_ad / norm_ad < 1 + EPSILON)   # (4,)
#     return cond1*cond2

# def box_in_box(corners1, corners2):
#     """check if corners of two boxes lie in each other
#     Args:
#         corners1 (np.array): 4x2 coordinates of corners
#         corners2 (np.array): 4x2 coordinates of corners
#     Returns:
#         c1_in_2: (4,) Bool. i-th corner of box1 in box2
#         c2_in_1: (4,) Bool. i-th corner of box2 in box1
#     """
#     c1_in_2 = box1_in_box2(corners1, corners2)
#     c2_in_1 = box1_in_box2(corners2, corners1)
#     return c1_in_2, c2_in_1

# def build_vertices(corners1, corners2):
#     """find all vertices of the polygon for intersection of 2 boxes
#     vertices include intersection points of edges and box corner in the other box
#     Args:
#         corners1 (np.array): 4x2 coordinates of corners
#         corners2 (np.array): 4x2 coordinates of corners
#     Returns:
#         poly_vertices (N, 2): vertices of polygon
#     """
    
#     c1_in_2, c2_in_1 = box_in_box(corners1, corners2)
#     corners_eff = np.concatenate([corners1[c1_in_2,:], corners2[c2_in_1,:]], axis=0)

#     inters, mask = box_intersection(corners1, corners2)
#     inters_lin = np.reshape(inters, (-1, 2))
#     mask_lin = np.reshape(mask, (-1, ))
#     inter_points = inters_lin[mask_lin, :]

#     poly_vertices = np.concatenate([corners_eff, inter_points], axis=0)
#     return poly_vertices

# def compare_vertices(v1, v2):
#     """compare two points according to the its angle around the origin point
#     of coordinate system. Useful for sorting vertices in anti-clockwise order
#     Args:
#         v1 (2, ): x1, y1
#         v2 (2, ): x2, y2
#     Returns:
#         int : 1 if angle1 > angle2. else -1
#     """
#     x1, y1 = v1
#     x2, y2 = v2
#     n1 = np.sqrt(x1*x1 + y1*y1) + EPSILON
#     n2 = np.sqrt(x2*x2 + y2*y2) + EPSILON
#     if y1 > 0 and y2 < 0:
#         return -1
#     elif y1 < 0 and y2 > 0:
#         return 1
#     elif y1 > 0 and y2 > 0:
#         if x1/n1 < x2/n2:
#             return 1
#         else:
#             return -1
#     else:
#         if x1/n1 > x2/n2:
#             return 1
#         else:
#             return -1

# def vertices2area(vertices):
#     """sort vertices in anti-clockwise order and calculate the area of polygon
#     Args:
#         vertices (N, 2) with N>2: vertices of a convex polygon
#     Returns:
#         area: area of polygon
#         ls: sorted vertices (normalized to centroid)
#     """
#     mean = np.mean(vertices, axis=0, keepdims=True)
#     vertices_normalized = vertices - mean
#     # sort vertices clockwise
#     ls = np.array(list(sorted(vertices_normalized, key=functools.cmp_to_key(compare_vertices))))
#     ls_ext = np.concatenate([ls, ls[0:1, :]], axis=0)
#     # formula of computing the area of polygon
#     total = ls_ext[0:-1, 0]*ls_ext[1:, 1] - ls_ext[1:, 0] * ls_ext[0:-1, 1]
#     total = np.sum(total)
#     area = np.abs(total) / 2
#     return area

# def box_intersection_area(corners1, corners2):
#     v = build_vertices(corners1, corners2)
#     if v.shape[0] < 3:
#         return 0
#     else:
#         return vertices2area(v)