import numpy as np

from utils.task1 import get_iou

def nms(pred, score, threshold):
    '''
    Task 5
    Implement NMS to reduce the number of predictions per frame with a threshold
    of 0.1. The IoU should be calculated only on the BEV.
    input
        pred (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
        score (N,) confidence scores
        threshold (float) upper bound threshold for NMS
    output
        s_f (M,7) 3D bounding boxes after NMS
        c_f (M,1) corresopnding confidence scores
    '''
    
    if pred.shape[0] == 0:
        return np.empty((0,7)), np.empty((0,1))
    
    s_f = []
    c_f = []

    # For calculation of IoU on 2D BEV, set (y,h)=(0,1)
    pred_copy = pred.copy()
    pred_copy[:,1] = 0 # y=0
    pred_copy[:,3] = 1 # h=1
    
    while pred.shape[0] > 0:
        # add proposal with highest confidence score in the remaining set
        i = np.argmax(score)
        Di = pred_copy[i:i+1]
        s_f.append(pred[i])
        c_f.append(score[i])

        # remove it from the remaining set
        pred = np.delete(pred, i, 0)
        score = np.delete(score, i)
        pred_copy = np.delete(pred_copy, i, 0)
        
        # remove highly overlapping predictions
        iou_mask = get_iou(Di, pred_copy)[0] < threshold
        score = score[iou_mask]
        pred = pred[iou_mask]
        pred_copy = pred_copy[iou_mask]

    s_f = np.array(s_f)
    c_f = np.array(c_f).reshape(-1,1)

    return s_f, c_f
