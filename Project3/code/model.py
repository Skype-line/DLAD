# Course: Deep Learning for Autonomous Driving, ETH Zurich
# Material for Project 3
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule, build_shared_mlp

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.__dict__.update(config)

        # xyz encoder
        if self.if_xyz_mlp:
            self.MLP_input_channel = 3
            self.xyz_CT_up_layer = build_shared_mlp([self.MLP_input_channel] + self.xyz_ct_mlp,
                                                    bn=False)
            if not self.config['if_bin_loss']:
                self.xyzn_up_layer = build_shared_mlp([4] + self.xyzn_mlp,
                                                    bn=False)
            c_out = self.xyz_ct_mlp[-1]
            if not self.config['if_bin_loss']:
                total_out = self.xyz_ct_mlp[-1] * 2 + self.xyzn_mlp[-1]
            else:
                total_out = self.xyz_ct_mlp[-1] * 2
            self.merge_down_layer = build_shared_mlp([total_out, c_out], bn=False)

        # Encoder
        channel_in = self.channel_in
        self.set_abstraction = nn.ModuleList()
        for k in range(len(self.npoint)):
            mlps = [channel_in] + self.mlps[k]
            npoint = self.npoint[k] if self.npoint[k]!=-1 else None
            self.set_abstraction.append(
                    PointnetSAModule(
                        npoint=npoint,
                        radius=self.radius[k],
                        nsample=self.nsample[k],
                        mlp=self.mlps[k],
                        use_xyz=True,
                        bn=False
                    )
                )
            channel_in = mlps[-1]

        # Classification head
        cls_layers = []
        pre_channel = channel_in
        for k in range(len(self.cls_fc)):
            cls_layers.extend([
                nn.Conv1d(pre_channel, self.cls_fc[k], kernel_size=1),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.cls_fc[k]
        cls_layers.extend([
            nn.Conv1d(pre_channel, 1, kernel_size=1),
            nn.Sigmoid()
        ])
        self.cls_layers = nn.Sequential(*cls_layers)

        # Regression head
        if self.if_bin_loss:
            per_loc_bin_num = int(self.LOC_SCOPE / self.LOC_BIN_SIZE) * 2
            loc_y_bin_num = int(self.LOC_Y_SCOPE / self.LOC_Y_BIN_SIZE) * 2
            reg_channel = per_loc_bin_num * 4 + self.NUM_HEAD_BIN * 2 + 3
            reg_channel += (1 if not self.LOC_Y_BY_BIN else loc_y_bin_num * 2)
            reg_layers = []
            pre_channel = channel_in
            for k in range(len(self.reg_fc)):
                reg_layers.extend([
                    nn.Conv1d(pre_channel, self.reg_fc[k], kernel_size=1),
                    nn.ReLU(inplace=True)
                ])
                pre_channel = self.reg_fc[k]
            reg_layers.append(nn.Conv1d(pre_channel, reg_channel, kernel_size=1))
            self.det_layers = nn.Sequential(*reg_layers)
        else:
            det_layers = []
            pre_channel = channel_in
            for k in range(len(self.reg_fc)):
                det_layers.extend([
                    nn.Conv1d(pre_channel, self.reg_fc[k], kernel_size=1),
                    nn.ReLU(inplace=True)
                ])
                pre_channel = self.reg_fc[k]
            det_layers.append(nn.Conv1d(pre_channel, 7, kernel_size=1))
            self.det_layers = nn.Sequential(*det_layers)



    def forward(self, x):
        xyz = x[..., 0:3].contiguous()                      # (B,N,3)    
        feat = x[..., 3:].transpose(1, 2).contiguous()      # (B,C,N)

        if self.if_xyz_mlp:
            xyz_CT_feature = self.xyz_CT_up_layer(x[..., 0:3].transpose(1, 2).unsqueeze(dim=3))
            if not self.config['if_bin_loss']:
                xyzn_feature = self.xyzn_up_layer(x[..., 3:7].transpose(1, 2).unsqueeze(dim=3))
                merged_feature = torch.cat((xyz_CT_feature, xyzn_feature, x[..., 7:].transpose(1, 2).unsqueeze(dim=3)), dim=1)
            else:
                merged_feature = torch.cat((xyz_CT_feature, x[..., 3:].transpose(1, 2).unsqueeze(dim=3)), dim=1)
            feat = self.merge_down_layer(merged_feature).squeeze(dim=3)
        for layer in self.set_abstraction:
            xyz, feat = layer(xyz, feat)
            
        # branch
        # xyz_cls, feat_cls = self.set_abstraction[2](xyz, feat)
        # xyz_det, feat_det = self.set_abstraction[3](xyz, feat)
        pred_class = self.cls_layers(feat).squeeze(dim=-1)  # (B,1)
        pred_box = self.det_layers(feat).squeeze(dim=-1)    # (B,7)
        return pred_box, pred_class