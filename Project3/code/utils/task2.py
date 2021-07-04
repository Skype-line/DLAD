import numpy as np
# from .task1 import label2corners
import math

# @profile
def roi_pool(pred, xyz, feat, config):
    '''
    Task 2
    a. Enlarge predicted 3D bounding boxes by delta=1.0 meters in all directions.
       As our inputs consist of coarse detection results from the stage-1 network,
       the second stage will benefit from the knowledge of surrounding points to
       better refine the initial prediction.
    b. Form ROI's by finding all points and their corresponding features that lie 
       in each enlarged bounding box. Each ROI should contain exactly 512 points.
       If there are more points within a bounding box, randomly sample until 512.
       If there are less points within a bounding box, randomly repeat points until
       512. If there are no points within a bounding box, the box should be discarded.
    input
        pred (K,7) bounding box labels
        xyz (N,3) point cloud
        feat (N,C) features     C=128+1(if USE_INTENSITY)+1(if USE_CT)
        config (dict) data config
    output
        valid_pred (K',7)
        pooled_xyz (K',M,3)
        pooled_feat (K',M,C)
            with K' indicating the number of valid bounding boxes that contain at least
            one point
    useful config hyperparameters
        config['delta'] extend the bounding box by delta on all sides (in meters)
        config['max_points'] number of points in the final sampled ROI
    '''
    delta = config['delta']
    max_points = config['max_points']

    K = pred.shape[0]

    # valid_pred = []
    pooled_xyz = []
    # pooled_feat = []

    range_x = pred[:,5]/2 + delta
    range_z = pred[:,4]/2 + delta
    max_dist = np.sqrt(range_x**2 + range_z**2)
    
    # define init condition
    cond_1 = np.arange(0, xyz.shape[0])

    # pre compute z_unrotate
    # ry = pred[:,-1].reshape((K, 1))
    x = xyz[:, 0]
    z = xyz[:, 2]
    y = xyz[:, 1]
    # z_unrotate_all = - np.sin(ry) * (x - pred[:, 0].reshape((K, 1))) + np.cos(ry) * (z -  pred[:, 2].reshape((K, 1)))
    conds = np.ones((K, max_points), dtype=np.uint64)
    idx = 0 # combined with conds, indicate the number of valid bounding box up to now
    # conds = []
    valid = []

    for i in range(K):
        # transform points to the box coordinate
        ry = pred[i,-1]
        bx = pred[i, 0]
        bz = pred[i, 2]
        by = pred[i, 1]
        bh = pred[i, 3]

        # init coarse filter to speedup selection
        cond = cond_1[abs(x - bx)+abs(z - bz)<1.415*max_dist[i]]
        
        # filter points in the box
        y_unrotate = y[cond] - by
        cond_y = abs(y_unrotate + bh/2)< bh/2 + delta
        cond = cond[cond_y]

        z_unrotate = math.sin(ry) * (x[cond] - bx) + math.cos(ry) * (z[cond] - bz)
        #z_unrotate = z_unrotate_all[i, :]
        cond_z = abs(z_unrotate) < range_z[i]
        cond = cond[cond_z]

        x_unrotate = math.cos(ry) * (x[cond] - bx) - math.sin(ry) * (z[cond] - bz)
        cond_x = abs(x_unrotate) < range_x[i]
        cond = cond[cond_x]
        
        # check validify
        N_pt = cond.shape[0]
        if N_pt == 0:
            conds = np.delete(conds, idx, 0)
            continue
        # # pooling
        elif N_pt > max_points:
            sample_mask = np.random.permutation(N_pt)[:max_points]
        else:
            sample_mask = np.concatenate((np.arange(N_pt), np.random.randint(0, N_pt, max_points-N_pt)))
            # sample_mask = np.concatenate((np.arange(N_pt), np.random.choice(N_pt, max_points-N_pt, replace=True)))
        
        cond = cond[sample_mask]
        conds[idx,:] = cond
        idx += 1
        # cond = cond[sample_mask]
        # conds.append(cond)
        valid.append(i) #+np.array([0, delta, 0, 2*delta, 2*delta, 2*delta, 0]))
        if config['if_xyz_mlp']:
            x_unrotate = math.cos(ry) * (x[cond] - bx) - math.sin(ry) * (z[cond] - bz)
            y_unrotate = y[cond] - by
            z_unrotate = math.sin(ry) * (x[cond] - bx) + math.cos(ry) * (z[cond] - bz)
            pooled_xyz.append(np.stack((x_unrotate,y_unrotate,z_unrotate), axis=-1))
    


    # assert len(valid) == conds.shape[0]
    # pooled_xyz = np.array([xyz[cond] for cond in conds])
    # pooled_feat = np.array([feat[cond] for cond in conds])
    # valid_pred = pred[valid]
    if config['if_xyz_mlp']:
        return pred[valid], np.array(pooled_xyz), feat[conds]
    else:    
        return pred[valid], xyz[conds], feat[conds]