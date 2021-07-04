import torch
import torch.nn.functional as F
from mtl.datasets.definitions import *

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p, SelfAttention, SqueezeAndExcitation


class ModelDeepLabV3PlusSASE(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        # get output channels for depth and seg
        ch_out_seg = outputs_desc[MOD_SEMSEG]
        ch_out_depth = outputs_desc[MOD_DEPTH]

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True, # modified to be true
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)
        ch_se = int(ch_out_encoder_bottleneck/4)

        self.aspp_seg = ASPP(ch_out_encoder_bottleneck, 256)
        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256)

        self.decoder_seg_1 = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out_seg)
        self.decoder_depth_1 = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out_depth)

        self.SA_seg = SelfAttention(256, 256)
        self.SA_depth = SelfAttention(256, 256)

        self.decoder_seg_2 = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out_seg)
        self.decoder_depth_2 = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out_depth)

        self.RA_seg = torch.nn.Conv2d(ch_se, ch_se, kernel_size=1, bias=False)
        self.RA_depth = torch.nn.Conv2d(ch_se, ch_se, kernel_size=1, bias=False)

        self.SE_seg = SqueezeAndExcitation(ch_out_encoder_bottleneck)
        self.SE_depth = SqueezeAndExcitation(ch_out_encoder_bottleneck)

        self.bn_seg_1 = torch.nn.BatchNorm2d(ch_se)
        self.bn_depth_1 = torch.nn.BatchNorm2d(ch_se)
        self.bn_seg_2 = torch.nn.BatchNorm2d(ch_se)
        self.bn_depth_2 = torch.nn.BatchNorm2d(ch_se)
        self.bn_seg_3 = torch.nn.BatchNorm2d(ch_out_encoder_bottleneck)
        self.bn_depth_3 = torch.nn.BatchNorm2d(ch_out_encoder_bottleneck)

        self.conv1 = torch.nn.Conv2d(ch_out_encoder_bottleneck, ch_se, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv2d(ch_se, ch_se, kernel_size=3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(ch_se, ch_out_encoder_bottleneck, kernel_size=1, stride=1, bias=False)

        self.relu = torch.nn.ReLU(inplace=True)


    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        # SE bottleneck
        bf_se_seg = self.conv1(features_lowest)
        bf_se_seg = self.bn_seg_1(bf_se_seg)
        bf_se_seg = self.relu(bf_se_seg)
        bf_se_depth = self.conv1(features_lowest)
        bf_se_depth = self.bn_depth_1(bf_se_depth)
        bf_se_depth = self.relu(bf_se_depth)

        bf_se_seg = self.conv2(bf_se_seg) + self.RA_seg(bf_se_seg)
        bf_se_seg = self.bn_seg_2(bf_se_seg)
        bf_se_seg = self.relu(bf_se_seg)
        bf_se_depth = self.conv2(bf_se_depth) + self.RA_depth(bf_se_depth)
        bf_se_depth = self.bn_depth_2(bf_se_depth)
        bf_se_depth = self.relu(bf_se_depth)

        bf_se_seg = self.conv3(bf_se_seg)
        bf_se_seg = self.bn_seg_3(bf_se_seg)
        bf_se_depth = self.conv3(bf_se_depth)
        bf_se_depth = self.bn_depth_3(bf_se_depth)

        feature_se_seg = self.SE_seg(bf_se_seg) + features_lowest
        feature_se_seg = self.relu(feature_se_seg)

        feature_se_depth = self.SE_depth(bf_se_depth) + features_lowest
        feature_se_depth = self.relu(feature_se_depth)

        # aspp
        features_seg = self.aspp_seg(feature_se_seg)

        features_depth = self.aspp_depth(feature_se_depth)

        # decoder 1
        predictions_4x_seg_1, final_features_seg_1 = self.decoder_seg_1(features_seg, features[4])

        predictions_4x_depth_1, final_features_depth_1 = self.decoder_depth_1(features_depth, features[4])

        # prediction 1
        predictions_1x_seg_1 = F.interpolate(predictions_4x_seg_1, size=input_resolution, mode='bilinear', align_corners=False)

        predictions_1x_depth_1 = F.interpolate(predictions_4x_depth_1, size=input_resolution, mode='bilinear', align_corners=False)
        

        # self attention
        sa_features_seg = self.SA_seg(final_features_seg_1)
        sa_features_depth = self.SA_depth(final_features_depth_1)

        # distillation
        features_input_seg_2 = final_features_seg_1 + sa_features_depth
        features_input_depth_2 = final_features_depth_1 + sa_features_seg

        # decoder 2
        predictions_4x_seg_2, _ = self.decoder_seg_2(features_input_seg_2, features[4])
        predictions_4x_depth_2, _ = self.decoder_depth_2(features_input_depth_2, features[4])

        # prediction 2
        predictions_1x_seg_2 = F.interpolate(predictions_4x_seg_2, size=input_resolution, mode='bilinear', align_corners=False)

        predictions_1x_depth_2 = F.interpolate(predictions_4x_depth_2, size=input_resolution, mode='bilinear', align_corners=False)

        out = {}

        out[MOD_SEMSEG] = [predictions_1x_seg_1, predictions_1x_seg_2]
        out[MOD_DEPTH] = [predictions_1x_depth_1, predictions_1x_depth_2]

        return out
