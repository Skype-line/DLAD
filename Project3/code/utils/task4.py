import torch
import torch.nn as nn
from utils.Rotated_IoU.oriented_iou_loss import cal_giou_3d
import numpy as np
import torch.nn.functional as F

class RegressionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''
        positive_mask = iou >= self.config['positive_reg_lb']
        pred_pos = pred[positive_mask]
        target_pos = target[positive_mask]
        if len(pred_pos)==0:
            total_loss = 0
        else:
            total_loss = self.loss(pred_pos[:,0:3], target_pos[:,0:3]) \
            + self.loss(pred_pos[:,3:6], target_pos[:,3:6]) * 3 + self.loss(pred_pos[:,-1], target_pos[:,-1])

        return total_loss

class ClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()

    def forward(self, pred, iou):
        '''
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,) predicted confidence score
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        '''
        pos_mask = iou >= self.config['positive_cls_lb']
        neg_mask = iou <= self.config['negative_cls_ub']
        valid_mask = pos_mask + neg_mask # logical OR
        iou[pos_mask] = 1
        iou[neg_mask] = 0
        
        return self.loss(pred[valid_mask,0], iou[valid_mask].float())


class RegressionIOULoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''
        positive_mask = iou >= self.config['positive_reg_lb']
        pred_pos = pred[positive_mask]
        target_pos = target[positive_mask]
        # B = self.config['data']['batch_size'] 
        B = 1
        pred_pos = pred_pos.cuda()
        target_pos = target_pos.cuda()
        
        if len(pred_pos)==0:
            total_loss = 0
        else:
            total_loss, iou3d = cal_giou_3d(pred_pos.view([B,-1,7]), target_pos.view([B,-1,7]))
    
        return total_loss


def get_reg_loss(pred_reg, reg_label, loc_scope, loc_bin_size, num_head_bin, anchor_size,
                 get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25, get_ry_fine=False):

    """
    Bin-based 3D bounding boxes regression loss. See https://arxiv.org/abs/1812.04244 for more details.
    
    :param pred_reg: (N, C)
    :param reg_label: (N, 7) [dx, dy, dz, h, w, l, ry]
    :param loc_scope: constant
    :param loc_bin_size: constant
    :param num_head_bin: constant
    :param anchor_size: (N, 3) or (3)
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

    reg_loss_dict = {}
    loc_loss = 0

    # xz localization loss
    x_offset_label, y_offset_label, z_offset_label = reg_label[:, 0], reg_label[:, 1], reg_label[:, 2]
    x_shift = torch.clamp(x_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
    z_shift = torch.clamp(z_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
    x_bin_label = (x_shift / loc_bin_size).floor().long()
    z_bin_label = (z_shift / loc_bin_size).floor().long()

    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r

    loss_x_bin = F.cross_entropy(pred_reg[:, x_bin_l: x_bin_r], x_bin_label)
    loss_z_bin = F.cross_entropy(pred_reg[:, z_bin_l: z_bin_r], z_bin_label)
    reg_loss_dict['loss_x_bin'] = loss_x_bin.item()
    reg_loss_dict['loss_z_bin'] = loss_z_bin.item()
    loc_loss += loss_x_bin + loss_z_bin

    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        x_res_label = x_shift - (x_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        z_res_label = z_shift - (z_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        x_res_norm_label = x_res_label / loc_bin_size
        z_res_norm_label = z_res_label / loc_bin_size

        x_bin_onehot = torch.cuda.FloatTensor(x_bin_label.size(0), per_loc_bin_num).zero_()
        x_bin_onehot.scatter_(1, x_bin_label.view(-1, 1).long(), 1)
        z_bin_onehot = torch.cuda.FloatTensor(z_bin_label.size(0), per_loc_bin_num).zero_()
        z_bin_onehot.scatter_(1, z_bin_label.view(-1, 1).long(), 1)

        loss_x_res = F.smooth_l1_loss((pred_reg[:, x_res_l: x_res_r] * x_bin_onehot).sum(dim=1), x_res_norm_label)
        loss_z_res = F.smooth_l1_loss((pred_reg[:, z_res_l: z_res_r] * z_bin_onehot).sum(dim=1), z_res_norm_label)
        reg_loss_dict['loss_x_res'] = loss_x_res.item()
        reg_loss_dict['loss_z_res'] = loss_z_res.item()
        loc_loss += loss_x_res + loss_z_res

    # y localization loss
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r

        y_shift = torch.clamp(y_offset_label + loc_y_scope, 0, loc_y_scope * 2 - 1e-3)
        y_bin_label = (y_shift / loc_y_bin_size).floor().long()
        y_res_label = y_shift - (y_bin_label.float() * loc_y_bin_size + loc_y_bin_size / 2)
        y_res_norm_label = y_res_label / loc_y_bin_size

        y_bin_onehot = torch.cuda.FloatTensor(y_bin_label.size(0), loc_y_bin_num).zero_()
        y_bin_onehot.scatter_(1, y_bin_label.view(-1, 1).long(), 1)

        loss_y_bin = F.cross_entropy(pred_reg[:, y_bin_l: y_bin_r], y_bin_label)
        loss_y_res = F.smooth_l1_loss((pred_reg[:, y_res_l: y_res_r] * y_bin_onehot).sum(dim=1), y_res_norm_label)

        reg_loss_dict['loss_y_bin'] = loss_y_bin.item()
        reg_loss_dict['loss_y_res'] = loss_y_res.item()

        loc_loss += loss_y_bin + loss_y_res
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        loss_y_offset = F.smooth_l1_loss(pred_reg[:, y_offset_l: y_offset_r].sum(dim=1), y_offset_label)
        reg_loss_dict['loss_y_offset'] = loss_y_offset.item()
        loc_loss += loss_y_offset

    # angle loss
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_label = reg_label[:, 6]

    if get_ry_fine:
        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin

        ry_label = ry_label % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
        ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        shift_angle = (ry_label + np.pi * 0.5) % (2 * np.pi)  # (0 ~ pi)

        shift_angle = torch.clamp(shift_angle - np.pi * 0.25, min=1e-3, max=np.pi * 0.5 - 1e-3)  # (0, pi/2)

        # bin center is (5, 10, 15, ..., 85)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    else:
        # divide 2pi into several bins
        angle_per_class = (2 * np.pi) / num_head_bin
        heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi

        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    ry_bin_onehot = torch.cuda.FloatTensor(ry_bin_label.size(0), num_head_bin).zero_()
    ry_bin_onehot.scatter_(1, ry_bin_label.view(-1, 1).long(), 1)
    loss_ry_bin = F.cross_entropy(pred_reg[:, ry_bin_l:ry_bin_r], ry_bin_label)
    loss_ry_res = F.smooth_l1_loss((pred_reg[:, ry_res_l: ry_res_r] * ry_bin_onehot).sum(dim=1), ry_res_norm_label)

    reg_loss_dict['loss_ry_bin'] = loss_ry_bin.item()
    reg_loss_dict['loss_ry_res'] = loss_ry_res.item()
    angle_loss = loss_ry_bin + loss_ry_res

    # size loss
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert pred_reg.shape[1] == size_res_r, '%d vs %d' % (pred_reg.shape[1], size_res_r)

    size_res_norm_label = (reg_label[:, 3:6] - anchor_size) / anchor_size
    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    size_loss = F.smooth_l1_loss(size_res_norm, size_res_norm_label)

    # Total regression loss
    reg_loss_dict['loss_loc'] = loc_loss
    reg_loss_dict['loss_angle'] = angle_loss
    reg_loss_dict['loss_size'] = size_loss

    return loc_loss, angle_loss, size_loss, reg_loss_dict


class RegressionBinLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''
        positive_mask = iou >= self.config['positive_reg_lb']
        pred_pos = pred[positive_mask]
        target_pos = target[positive_mask]
        # B = self.config['data']['batch_size'] 
        
        if len(pred_pos)==0:
            total_loss = 0
        else:
            anchor_size = pred_pos[:, 3:6]
            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                get_reg_loss(pred_pos,
                            target_pos,
                            loc_scope=self.config['LOC_SCOPE'],
                            loc_bin_size=self.config['LOC_BIN_SIZE'],
                            num_head_bin=self.config['NUM_HEAD_BIN'],
                            anchor_size=anchor_size,
                            get_xz_fine=True,
                            get_y_by_bin=self.config['LOC_Y_BY_BIN'],
                            loc_y_scope=self.config['LOC_Y_SCOPE'], loc_y_bin_size=self.config['LOC_Y_BIN_SIZE'],
                            get_ry_fine=True)

            loss_size = 3 * loss_size  # consistent with old codes
            total_loss = loss_loc + loss_angle + loss_size
    
        return total_loss


def rotate_pc_along_y_torch(pc, rot_angle):
    """
    :param pc: (N, 3 + C)
    :param rot_angle: (N)
    :return:
    """
    cosa = torch.cos(rot_angle).view(-1, 1)
    sina = torch.sin(rot_angle).view(-1, 1)

    raw_1 = torch.cat([cosa, -sina], dim=1)
    raw_2 = torch.cat([sina, cosa], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)  # (N, 2, 2)

    pc_temp = pc[:, [0, 2]].unsqueeze(dim=1)  # (N, 1, 2)

    pc[:, [0, 2]] = torch.matmul(pc_temp, R.permute(0, 2, 1)).squeeze(dim=1)
    return pc


def decode_bbox_target(roi_box3d, pred_reg, loc_scope, loc_bin_size, num_head_bin, anchor_size,
                       get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25, get_ry_fine=False):
    """
    :param roi_box3d: (N, 7)
    :param pred_reg: (N, C)
    :param loc_scope:
    :param loc_bin_size:
    :param num_head_bin:
    :param anchor_size:
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """
    anchor_size = anchor_size.to(roi_box3d.get_device())
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

    # recover xz localization
    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r

    x_bin = torch.argmax(pred_reg[:, x_bin_l: x_bin_r], dim=1)
    z_bin = torch.argmax(pred_reg[:, z_bin_l: z_bin_r], dim=1)

    pos_x = x_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope
    pos_z = z_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope

    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        x_res_norm = torch.gather(pred_reg[:, x_res_l: x_res_r], dim=1, index=x_bin.unsqueeze(dim=1)).squeeze(dim=1)
        z_res_norm = torch.gather(pred_reg[:, z_res_l: z_res_r], dim=1, index=z_bin.unsqueeze(dim=1)).squeeze(dim=1)
        x_res = x_res_norm * loc_bin_size
        z_res = z_res_norm * loc_bin_size

        pos_x += x_res
        pos_z += z_res

    # recover y localization
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r

        y_bin = torch.argmax(pred_reg[:, y_bin_l: y_bin_r], dim=1)
        y_res_norm = torch.gather(pred_reg[:, y_res_l: y_res_r], dim=1, index=y_bin.unsqueeze(dim=1)).squeeze(dim=1)
        y_res = y_res_norm * loc_y_bin_size
        pos_y = y_bin.float() * loc_y_bin_size + loc_y_bin_size / 2 - loc_y_scope + y_res
        pos_y = pos_y + roi_box3d[:, 1]
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        pos_y = roi_box3d[:, 1] + pred_reg[:, y_offset_l]

    # recover ry rotation
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_bin = torch.argmax(pred_reg[:, ry_bin_l: ry_bin_r], dim=1)
    ry_res_norm = torch.gather(pred_reg[:, ry_res_l: ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)
    if get_ry_fine:
        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        ry = (ry_bin.float() * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4
    else:
        angle_per_class = (2 * np.pi) / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)

        # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
        ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
        ry[ry > np.pi] -= 2 * np.pi

    # recover size
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert size_res_r == pred_reg.shape[1]

    size_res_norm = pred_reg[:, size_res_l: size_res_r]
    hwl = size_res_norm * anchor_size + anchor_size

    # shift to original coords
    roi_center = roi_box3d[:, 0:3]
    shift_ret_box3d = torch.cat((pos_x.view(-1, 1), pos_y.view(-1, 1), pos_z.view(-1, 1), hwl, ry.view(-1, 1)), dim=1)
    ret_box3d = shift_ret_box3d
    if roi_box3d.shape[1] == 7:
        roi_ry = roi_box3d[:, 6]
        ret_box3d = rotate_pc_along_y_torch(shift_ret_box3d, - roi_ry)
        ret_box3d[:, 6] += roi_ry
    ret_box3d[:, [0, 2]] += roi_center[:, [0, 2]]

    return ret_box3d
