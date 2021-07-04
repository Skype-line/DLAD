import torch
import torch.nn.functional as F
from mtl.datasets.definitions import *

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, ASPP_SASE, DecoderDeeplabV3p, DecoderDeeplabV3pUpConv, SelfAttention, UpProject


class ModelDeepLabV3PlusUpConv(torch.nn.Module):
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

        self.aspp_seg = ASPP(ch_out_encoder_bottleneck, 256)
        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256)

        self.decoder_seg_1 = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out_seg)
        self.decoder_depth_1 = DecoderDeeplabV3pUpConv(256, ch_out_encoder_4x, ch_out_depth, 1)

        self.SA_seg = SelfAttention(512, 256)
        self.SA_depth = SelfAttention(512, 256)

        self.decoder_seg_2 = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out_seg)
        self.decoder_depth_2 = DecoderDeeplabV3pUpConv(256, ch_out_encoder_4x, ch_out_depth, 0)


    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        # aspp
        features_seg = self.aspp_seg(features_lowest)

        features_depth = self.aspp_depth(features_lowest)

        # decoder 1
        predictions_2x_seg_1, final_features_seg_1 = self.decoder_seg_1(features_seg, features[4])

        predictions_2x_depth_1, final_features_depth_1 = self.decoder_depth_1(features_depth, features[4])

        # prediction 1
        predictions_1x_seg_1 = F.interpolate(predictions_2x_seg_1, size=input_resolution, mode='bilinear', align_corners=False)
        
        predictions_1x_depth_1 = F.interpolate(predictions_2x_depth_1, size=input_resolution, mode='bilinear', align_corners=False)

        # self attention
        sa_features_seg = self.SA_seg(torch.concat([final_features_seg_1,final_features_depth_1], dim=1))
        sa_features_depth = self.SA_depth(torch.concat([final_features_seg_1,final_features_depth_1], dim=1))

        # distillation
        features_input_seg_2 = final_features_seg_1 + sa_features_depth
        features_input_depth_2 = final_features_depth_1 + sa_features_seg

        # decoder 2
        predictions_2x_seg_2, _ = self.decoder_seg_2(features_input_seg_2, features[4])
        predictions_2x_depth_2, _ = self.decoder_depth_2(features_input_depth_2, features[4])

        # prediction 2
        predictions_1x_seg_2 = F.interpolate(predictions_2x_seg_2, size=input_resolution, mode='bilinear', align_corners=False)

        predictions_1x_depth_2 = F.interpolate(predictions_2x_depth_2, size=input_resolution, mode='bilinear', align_corners=False)

        out = {}

        out[MOD_SEMSEG] = [predictions_1x_seg_1, predictions_1x_seg_2]
        out[MOD_DEPTH] = [predictions_1x_depth_1, predictions_1x_depth_2]

        return out
