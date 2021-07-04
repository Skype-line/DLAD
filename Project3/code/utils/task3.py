import numpy as np

from .task1 import get_iou

def sample_proposals(pred, target, xyz, feat, config, train=False):
    '''
    Task 3
    a. Using the highest IoU, assign each proposal a ground truth annotation. For each assignment also
       return the IoU as this will be required later on.
    b. Sample 64 proposals per scene. If the scene contains at least one foreground and one background
       proposal, of the 64 samples, at most 32 should be foreground proposals. Otherwise, all 64 samples
       can be either foreground or background. If there are less background proposals than 32, existing
       ones can be repeated.
       Furthermore, of the sampled background proposals, 50% should be easy samples and 50% should be
       hard samples when both exist within the scene (again, can be repeated to pad up to equal samples
       each). If only one difficulty class exists, all samples should be of that class.
    input
        pred (N,7) predicted bounding box labels
        target (M,7) ground truth bounding box labels
        xyz (N,512,3) pooled point cloud
        feat (N,512,C) pooled features
        config (dict) data config containing thresholds
        train (string) True if training
    output
        assigned_targets (64,7) target box for each prediction based on highest iou
        xyz (64,512,3) indices 
        feat (64,512,C) indices
        iou (64,) iou of each prediction and its assigned target box
    useful config hyperparameters
        config['t_bg_hard_lb'] threshold background lower bound for hard difficulty
        config['t_bg_up'] threshold background upper bound
        config['t_fg_lb'] threshold foreground lower bound
        config['num_fg_sample'] maximum allowed number of foreground samples
        config['bg_hard_ratio'] background hard difficulty ratio (#hard samples/ #background samples)
    '''

    num_samples = config['num_samples']
    num_fg_sample = config['num_fg_sample']
    bg_hard_ratio = config['bg_hard_ratio']

    iou = get_iou(pred, target) # (N,M)
    N, M = iou.shape
    pred_id_1 = np.arange(N)
    targ_id_1 = np.argmax(iou, axis=1)
    high_iou_1 = np.max(iou, axis=1)

    if train == False:
        return target[targ_id_1], xyz, feat, high_iou_1
    else:
        valid_mask = (high_iou_1 < config['t_bg_up']) + (high_iou_1 >=config['t_fg_lb']) # OR
        pred_targ_iou_1 = np.stack((pred_id_1[valid_mask], targ_id_1[valid_mask], high_iou_1[valid_mask]),axis=-1)

        pred_id_2 = np.argmax(iou, axis=0)
        targ_id_2 = np.arange(M)
        high_iou_2 = np.max(iou, axis=0)
        pred_targ_iou_2 = np.stack((pred_id_2, targ_id_2, high_iou_2), axis=-1)

        easy_bg = pred_targ_iou_1[pred_targ_iou_1[:,-1] < config['t_bg_hard_lb']]
        hard_bg = pred_targ_iou_1[(pred_targ_iou_1[:,-1] >= config['t_bg_hard_lb']) * (pred_targ_iou_1[:,-1] < config['t_bg_up'])]
        fg_1 =  pred_targ_iou_1[pred_targ_iou_1[:,-1] >= config['t_fg_lb']]
        fg = np.concatenate((fg_1, pred_targ_iou_2), axis=0)

        num_easy_bg = easy_bg.shape[0]
        num_hard_bg = hard_bg.shape[0]
        num_fg = fg.shape[0]
        # print(num_easy_bg)
        # print(num_hard_bg)
        # print(num_fg)
    
        # pred_targ_iou_pair (64, 3)
        if num_easy_bg + num_hard_bg == 0:
            # sample_mask = np.random.choice(num_fg, num_samples, replace=True)
            sample_mask = random_sample(num_fg, num_samples)
            pred_targ_iou_pair = fg[sample_mask,:]
        elif num_fg == 0:
            pred_targ_iou_pair = bg_sample_proposals(num_samples, bg_hard_ratio, num_easy_bg, num_hard_bg, easy_bg, hard_bg)
        elif num_fg > num_fg_sample:
            # fg_sample_mask = np.random.choice(num_fg, num_fg_sample, replace=True)
            fg_sample_mask = random_sample(num_fg, num_fg_sample)
            fg_pred_targ_iou_pair = fg[fg_sample_mask,:]
            bg_pred_targ_iou_pair = bg_sample_proposals(num_samples-num_fg_sample, bg_hard_ratio, num_easy_bg, num_hard_bg, easy_bg, hard_bg)
            
            pred_targ_iou_pair = np.concatenate((fg_pred_targ_iou_pair, bg_pred_targ_iou_pair), axis=0)
        else:
            bg_pred_targ_iou_pair = bg_sample_proposals(num_samples-num_fg, bg_hard_ratio, num_easy_bg, num_hard_bg, easy_bg, hard_bg)
            pred_targ_iou_pair = np.concatenate((fg, bg_pred_targ_iou_pair), axis=0) 

        assigned_targets = target[pred_targ_iou_pair[:,1].astype(int),:]
        xyz = xyz[pred_targ_iou_pair[:,0].astype(int),:,:]
        feat = feat[pred_targ_iou_pair[:,0].astype(int),:,:]
        iou = pred_targ_iou_pair[:,2]
        assigned_preds = pred[pred_targ_iou_pair[:,0].astype(int),:]
        if config['if_bin_loss']:
            roi_center = assigned_preds[:, 0:3]
            roi_ry = assigned_preds[:, -1] % (2 * np.pi)
            center_target = assigned_targets[:, 0:3] - roi_center
            rotated_x = center_target[:, 0] * np.cos(roi_ry) - center_target[:, 2] * np.sin(roi_ry)
            rotated_z = center_target[:, 0] * np.sin(roi_ry) + center_target[:, 2] * np.cos(roi_ry)
            assigned_targets[:, 0] = rotated_x
            assigned_targets[:, 2] = rotated_z
            assigned_targets[:, 1] = center_target[:, 1]
            assigned_targets[:, 6] -= roi_ry

            return assigned_preds, assigned_targets, xyz, feat, iou

        return assigned_targets, xyz, feat, iou#, assigned_preds


def bg_sample_proposals(num_needed, bg_hard_ratio, num_easy_bg, num_hard_bg, easy_bg, hard_bg):
    if num_easy_bg == 0:
        # sample_mask = np.random.choice(num_hard_bg, num_needed, replace=True)
        sample_mask = random_sample(num_hard_bg, num_needed)
        pred_targ_iou_pair = hard_bg[sample_mask,:]
    elif num_hard_bg == 0:
        # sample_mask = np.random.choice(num_easy_bg, num_needed, replace=True)
        sample_mask = random_sample(num_easy_bg, num_needed)
        pred_targ_iou_pair = easy_bg[sample_mask,:]
    else:
        num_needed_hard = int(num_needed * bg_hard_ratio)
        num_needed_easy = num_needed - num_needed_hard
        
        # sample_mask_hard = np.random.choice(num_hard_bg, num_needed_hard, replace=True)
        sample_mask_hard = random_sample(num_hard_bg, num_needed_hard)
        pred_targ_iou_pair_hard = hard_bg[sample_mask_hard,:]

        # sample_mask_easy = np.random.choice(num_easy_bg, num_needed_easy, replace=True)
        sample_mask_easy = random_sample(num_easy_bg, num_needed_easy)
        pred_targ_iou_pair_easy = easy_bg[sample_mask_easy,:]

        pred_targ_iou_pair = np.concatenate((pred_targ_iou_pair_hard, pred_targ_iou_pair_easy), axis=0)
    
    return pred_targ_iou_pair


def random_sample(num_have, num_need):
    if num_have > num_need:
        sample_mask = np.random.permutation(num_have)[:num_need]
    else:
        sample_mask = np.concatenate((np.arange(num_have), np.random.randint(0, num_have, num_need-num_have)))
    
    return sample_mask