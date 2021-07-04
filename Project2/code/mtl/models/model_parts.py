import torch
import torch.nn.functional as F
import torchvision.models.resnet as resnet
import collections


class BasicBlockWithDilation(torch.nn.Module):
    """Workaround for prohibited dilation in BasicBlock in 0.4.0"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDilation, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


_basic_block_layers = {
    'resnet18': (2, 2, 2, 2),
    'resnet34': (3, 4, 6, 3),
}


def get_encoder_channel_counts(encoder_name):
    is_basic_block = encoder_name in _basic_block_layers
    ch_out_encoder_bottleneck = 512 if is_basic_block else 2048
    ch_out_encoder_4x = 64 if is_basic_block else 256
    return ch_out_encoder_bottleneck, ch_out_encoder_4x


class Encoder(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            pretrained = encoder_kwargs.pop('pretrained', False)
            progress = encoder_kwargs.pop('progress', True)
            model = resnet._resnet(
                name, BasicBlockWithDilation, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            )
        replace_stride_with_dilation = encoder_kwargs.get('replace_stride_with_dilation', (False, False, False))
        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        out = {1: x}
        sz_in = x.shape[3]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, sz_in) #2

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, sz_in) #4 (x4) ch: 64

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in) #8 ch: 64

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in) #16 ch: 128

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in) #32 ch: 256

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in) #32 if dilation ch: 512

        return out


class DecoderDeeplabV3p(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3p, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following
        self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=1, stride=1)
        
        # 1*1 conv according to the paper, 48 channels has best performance
        self.skip_to_reduced = torch.nn.Conv2d(skip_4x_ch, 48, kernel_size=1, stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(48)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv3x3_refine_1 = torch.nn.Conv2d(48 + bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(bottleneck_ch)
        self.conv3x3_refine_2 = torch.nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(bottleneck_ch)

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        
        # 1x1 conv
        features_skip_reduced = self.skip_to_reduced(features_skip_4x)
        features_skip_reduced = self.bn1(features_skip_reduced)
        features_skip_reduced = self.relu(features_skip_reduced)
        # upsampling
        features_bottleneck_4x = F.interpolate(
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )
        # concat
        concat_features = torch.cat((features_skip_reduced, features_bottleneck_4x),dim=1)
        # refine 1
        features_4x = self.conv3x3_refine_1(concat_features)
        features_4x = self.bn2(features_4x)
        features_4x = self.relu(features_4x)
        # refine 2
        features_4x = self.conv3x3_refine_2(features_4x)
        features_4x = self.bn3(features_4x)
        features_4x = self.relu(features_4x)
        # predict
        predictions_4x = self.features_to_predictions(features_4x)
        return predictions_4x, features_4x


class ASPPpart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )


class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rates=(3, 6, 9)):
        super().__init__()
        # TODO: Implement ASPP properly instead of the following
        self.conv_out = ASPPpart(5*out_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

        # layers
        self.conv1x1 = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv3x3_1 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0])
        self.conv3x3_2 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1])
        self.conv3x3_3 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2])
        self.image_pooling = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False), 
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )

    def forward(self, x):
        # TODO: Implement ASPP properly instead of the following
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        x5 = self.image_pooling(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        concat_x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.conv_out(concat_x)
        return out


class ASPP_SASE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rates=(3, 6, 9)):
        super().__init__()
        # TODO: Implement ASPP properly instead of the following
        self.conv_out = ASPPpart(5*out_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

        # layers
        self.conv1x1 = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv3x3_1 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0])
        self.conv3x3_2 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1])
        self.conv3x3_3 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2])
        self.image_pooling = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False), 
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.sa1x1 = SelfAttention(out_channels, out_channels)
        self.sa3x3_1 = SelfAttention(out_channels, out_channels)
        self.sa3x3_2 = SelfAttention(out_channels, out_channels)
        self.sa3x3_3 = SelfAttention(out_channels, out_channels)
        self.sa_pooling = SelfAttention(out_channels, out_channels)

        self.se = SqueezeAndExcitation(out_channels*5)

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        # TODO: Implement ASPP properly instead of the following
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        x5 = self.image_pooling(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        concat_x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # add scalewise attension
        ax1 = self.sa1x1(x1)
        ax2 = self.sa3x3_1(x2)
        ax3 = self.sa3x3_2(x3)
        ax4 = self.sa3x3_3(x4)
        ax5 = self.sa_pooling(x5)
        concat_ax = torch.cat((ax1, ax2, ax3, ax4, ax5), dim=1)
        out = concat_x + concat_ax
        out = self.relu(out)

        # squeeze and excitation
        out = self.se(out) + out
        out = self.relu(out)

        out = self.conv_out(out)

        return out


class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.attention = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.attention.weight.copy_(torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask


class SqueezeAndExcitation(torch.nn.Module):
    """
    Squeeze and excitation module as explained in https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, channels, r=16):
        super(SqueezeAndExcitation, self).__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // r),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // r, channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        squeezed = torch.mean(x, dim=(2, 3)).reshape(N, C)
        squeezed = self.transform(squeezed).reshape(N, C, 1, 1)
        return x * squeezed


class RASE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        ch_se = in_channels//4
        self.RA = torch.nn.Conv2d(ch_se, ch_se, kernel_size=1, bias=False)

        self.SE = SqueezeAndExcitation(out_channels)

        self.bn_1 = torch.nn.BatchNorm2d(ch_se)
        self.bn_2 = torch.nn.BatchNorm2d(ch_se)
        self.bn_3 = torch.nn.BatchNorm2d(out_channels)

        self.conv1 = torch.nn.Conv2d(in_channels, ch_se, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv2d(ch_se, ch_se, kernel_size=3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(ch_se, out_channels, kernel_size=1, stride=1, bias=False)

        self.relu = torch.nn.ReLU()

    def forward(self, input_feature):

        x = self.conv1(input_feature)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.conv2(x) + self.RA(x)
        x = self.bn_2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn_3(x)

        x = self.SE(x) + input_feature
        output = self.relu(x)
        
        return output


class RASE_simple(torch.nn.Module):
    def __init__(self, in_channels):
        super(RASE_simple, self).__init__()
        self.RA = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        self.SE = SqueezeAndExcitation(in_channels)

        self.bn = torch.nn.BatchNorm2d(in_channels)

        self.relu = torch.nn.ReLU()

    def forward(self, input_feature):

        x = input_feature + self.RA(input_feature)
        x = self.bn(x)

        x = self.SE(x) + input_feature
        output = self.relu(x)
        
        return output


class UpProject(torch.nn.Module):

    def __init__(self, in_channel, out_channel):
        super(UpProject, self).__init__()

        self.conv1_ = torch.nn.Sequential(collections.OrderedDict([
            ('conv1', torch.nn.Conv2d(in_channel, out_channel, kernel_size=3)),
            ('bn1', torch.nn.BatchNorm2d(out_channel)),
        ]))

        self.conv2_ = torch.nn.Sequential(collections.OrderedDict([
            ('conv1', torch.nn.Conv2d(in_channel, out_channel, kernel_size=(2, 3))),
            ('bn1', torch.nn.BatchNorm2d(out_channel)),
        ]))

        self.conv3_ = torch.nn.Sequential(collections.OrderedDict([
            ('conv1', torch.nn.Conv2d(in_channel, out_channel, kernel_size=(3, 2))),
            ('bn1', torch.nn.BatchNorm2d(out_channel)),
        ]))

        self.conv4_ = torch.nn.Sequential(collections.OrderedDict([
            ('conv1', torch.nn.Conv2d(in_channel, out_channel, kernel_size=2)),
            ('bn1', torch.nn.BatchNorm2d(out_channel)),
        ]))

        self.ps = torch.nn.PixelShuffle(2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        # print('Upmodule x size = ', x.size())
        x1 = self.conv1_(torch.nn.functional.pad(x, (1, 1, 1, 1)))
        x2 = self.conv2_(torch.nn.functional.pad(x, (1, 1, 0, 1)))
        x3 = self.conv3_(torch.nn.functional.pad(x, (0, 1, 1, 1)))
        x4 = self.conv4_(torch.nn.functional.pad(x, (0, 1, 0, 1)))

        x = torch.cat((x1, x2, x3, x4), dim=1)

        output = self.ps(x)
        output = self.relu(output)

        return output



class DecoderDeeplabV3pFullUpConv(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3pFullUpConv, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following
        self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch // 8, num_out_ch, kernel_size=1, stride=1)
        
        # 1*1 conv according to the paper, 48 channels has best performance
        self.skip_to_reduced = torch.nn.Conv2d(skip_4x_ch, 48, kernel_size=1, stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(48)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv3x3_refine_1 = torch.nn.Conv2d(48 + bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(bottleneck_ch)

        self.upconv1 = UpProject(bottleneck_ch, bottleneck_ch)
        self.upconv2 = UpProject(bottleneck_ch, bottleneck_ch)
        
        self.upconv_refine_1 = UpProject(48 + bottleneck_ch, bottleneck_ch // 4)
        self.upconv_refine_2 = UpProject(bottleneck_ch // 4, bottleneck_ch // 8)


    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        
        # 1x1 conv
        features_skip_reduced = self.skip_to_reduced(features_skip_4x)
        features_skip_reduced = self.bn1(features_skip_reduced)
        features_skip_reduced = self.relu(features_skip_reduced)
        # upsampling
        # features_bottleneck_4x = F.interpolate(
        #     features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        # )
        features_bottleneck_8x = self.upconv1(features_bottleneck)
        features_bottleneck_4x = self.upconv2(features_bottleneck_8x)

        # concat
        concat_features = torch.cat((features_skip_reduced, features_bottleneck_4x),dim=1)

        features_2x = self.upconv_refine_1(concat_features)
        features_1x = self.upconv_refine_2(features_2x)
        
        # predict
        predictions_1x = self.features_to_predictions(features_1x)
        return predictions_1x, features_1x


class DecoderDeeplabV3pUpConv(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch, if_up_bottleneck):
        super(DecoderDeeplabV3pUpConv, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following
        self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=1, stride=1)
        
        # 1*1 conv according to the paper, 48 channels has best performance
        self.skip_to_reduced = torch.nn.Conv2d(skip_4x_ch, 48, kernel_size=1, stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(48)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv3x3_refine_1 = torch.nn.Conv2d(48 + bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(bottleneck_ch)

        self.if_up_bottleneck = if_up_bottleneck

        if if_up_bottleneck:
            self.upconv1 = UpProject(bottleneck_ch, bottleneck_ch)
            self.upconv2 = UpProject(bottleneck_ch, bottleneck_ch)
            
        self.upconv_refine_1 = UpProject(bottleneck_ch, bottleneck_ch)
        
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        
        # 1x1 conv
        features_skip_reduced = self.skip_to_reduced(features_skip_4x)
        features_skip_reduced = self.bn1(features_skip_reduced)
        features_skip_reduced = self.relu(features_skip_reduced)
        # upsampling
        # features_bottleneck_4x = F.interpolate(
        #     features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        # )
        if self.if_up_bottleneck:
            features_bottleneck_8x = self.upconv1(features_bottleneck)
            features_bottleneck_4x = self.upconv2(features_bottleneck_8x)
        else:
            features_bottleneck_4x = features_bottleneck

        # concat
        concat_features = torch.cat((features_skip_reduced, features_bottleneck_4x),dim=1)
        features_4x = self.conv3x3_refine_1(concat_features)
        features_4x = self.bn2(features_4x)
        features_4x = self.relu(features_4x)
        
        # refine 2
        features_2x = self.upconv_refine_1(features_4x)
        features_4x = self.maxpool(features_2x)
        
        # predict
        predictions_2x = self.features_to_predictions(features_2x)
        return predictions_2x, features_4x


class DecoderDeeplabV3pFinal(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3pFinal, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following
        self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=1, stride=1)
        
        # 1*1 conv according to the paper, 48 channels has best performance
        self.skip_to_reduced = torch.nn.Conv2d(skip_4x_ch, 48, kernel_size=1, stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(48)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv3x3_refine_1 = torch.nn.Conv2d(48 + bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(bottleneck_ch)

        # consider different input shape
        self.upconv1 = UpProject(48, 48)
        self.upconv2 = UpProject(48, 48)


    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        
        # 1x1 conv
        features_skip_reduced = self.skip_to_reduced(features_skip_4x)
        features_skip_reduced = self.bn1(features_skip_reduced)
        features_skip_reduced = self.relu(features_skip_reduced)
        features_skip_reduced = self.upconv1(features_skip_reduced)
        features_skip_reduced = self.upconv2(features_skip_reduced)
        # upsampling
        # features_bottleneck_4x = F.interpolate(
        #     features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        # )
        features_bottleneck_1x = features_bottleneck

        # concat
        concat_features = torch.cat((features_skip_reduced, features_bottleneck_1x),dim=1)
        
        # refine 2
        features_1x = self.conv3x3_refine_1(concat_features)
        features_1x = self.bn2(features_1x)
        features_1x = self.relu(features_1x)

        # predict
        predictions_1x = self.features_to_predictions(features_1x)
        return predictions_1x, features_1x


class DecoderDeeplabV3pConv2xSkip(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, skip_2x_ch, num_out_ch, feature_out_ch, if_up_bottleneck):
        super(DecoderDeeplabV3pConv2xSkip, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following
        self.features_to_predictions = torch.nn.Conv2d(feature_out_ch, num_out_ch, kernel_size=1, stride=1)
        
        # 1*1 conv according to the paper, 48 channels has best performance
        self.skip4x_to_reduced = torch.nn.Conv2d(skip_4x_ch, 48, kernel_size=1, stride=1, bias=False)
        self.bn4x = torch.nn.BatchNorm2d(48)
        self.relu = torch.nn.ReLU()
        self.conv3x3_refine_4x = torch.nn.Conv2d(48 + bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_refine_4x = torch.nn.BatchNorm2d(bottleneck_ch)

        self.skip2x_to_reduced = torch.nn.Conv2d(skip_2x_ch, 24, kernel_size=1, stride=1, bias=False)
        self.bn2x = torch.nn.BatchNorm2d(24)
        self.relu = torch.nn.ReLU()
        self.conv3x3_refine_2x = torch.nn.Conv2d(24 + feature_out_ch, feature_out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_refine_2x = torch.nn.BatchNorm2d(feature_out_ch)

        self.if_up_bottleneck = if_up_bottleneck

        # consider different input shape
        # 
        if self.if_up_bottleneck:
            self.upconv_bottle1 = UpProject(bottleneck_ch, bottleneck_ch)
            self.upconv_bottle2 = UpProject(bottleneck_ch, bottleneck_ch)

        self.upconv4x = UpProject(bottleneck_ch, feature_out_ch)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, features_bottleneck, features_skip_4x, features_skip_2x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        
        # skip
        features_skip_4x_reduced = self.skip4x_to_reduced(features_skip_4x)
        features_skip_4x_reduced = self.bn4x(features_skip_4x_reduced)
        features_skip_4x_reduced = self.relu(features_skip_4x_reduced)

        features_skip_2x_reduced = self.skip2x_to_reduced(features_skip_2x)
        features_skip_2x_reduced = self.bn2x(features_skip_2x_reduced)
        features_skip_2x_reduced = self.relu(features_skip_2x_reduced)
        

        # upsampling
        if self.if_up_bottleneck:
            features_bottleneck_4x = self.upconv_bottle1(features_bottleneck)
            features_bottleneck_4x = self.upconv_bottle2(features_bottleneck_4x)
        else: 
            features_bottleneck_4x = F.interpolate(
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )

        # concat
        concat_features_4x = torch.cat((features_skip_4x_reduced, features_bottleneck_4x),dim=1)
        
        # refine 1
        features_4x = self.conv3x3_refine_4x(concat_features_4x)
        features_4x = self.bn_refine_4x(features_4x)
        features_4x = self.relu(features_4x)

        #upconv
        features_2x = self.upconv4x(features_4x)
        concat_features_2x = torch.cat((features_skip_2x_reduced, features_2x),dim=1)

        # refine2
        features_2x = self.conv3x3_refine_2x(concat_features_2x)
        features_2x = self.bn_refine_2x(features_2x)
        features_2x = self.relu(features_2x)

        features_4x_out = self.maxpool(features_2x)

        # predict
        predictions_2x = self.features_to_predictions(features_2x)
        return predictions_2x, features_4x_out



class DecoderDeeplabV3pConv2xSkip2x(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, skip_2x_ch, num_out_ch, feature_out_ch, if_up_bottleneck):
        super(DecoderDeeplabV3pConv2xSkip2x, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following
        self.features_to_predictions = torch.nn.Conv2d(feature_out_ch, num_out_ch, kernel_size=1, stride=1)
        
        # 1*1 conv according to the paper, 48 channels has best performance
        self.skip4x_to_reduced = torch.nn.Conv2d(skip_4x_ch, 48, kernel_size=1, stride=1, bias=False)
        self.bn4x = torch.nn.BatchNorm2d(48)
        self.relu = torch.nn.ReLU()
        self.conv3x3_refine_4x = torch.nn.Conv2d(48 + bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_refine_4x = torch.nn.BatchNorm2d(bottleneck_ch)

        self.skip2x_to_reduced = torch.nn.Conv2d(skip_2x_ch, 24, kernel_size=1, stride=1, bias=False)
        self.bn2x = torch.nn.BatchNorm2d(24)
        self.relu = torch.nn.ReLU()
        self.conv3x3_refine_2x = torch.nn.Conv2d(24 + feature_out_ch, feature_out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_refine_2x = torch.nn.BatchNorm2d(feature_out_ch)

        self.if_up_bottleneck = if_up_bottleneck

        # consider different input shape
        # 
        if self.if_up_bottleneck:
            self.upconv_bottle1 = UpProject(bottleneck_ch, bottleneck_ch)
            self.upconv_bottle2 = UpProject(bottleneck_ch, bottleneck_ch)

        self.upconv4x = UpProject(bottleneck_ch, feature_out_ch)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, features_bottleneck, features_skip_4x, features_skip_2x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        
        # skip
        features_skip_4x_reduced = self.skip4x_to_reduced(features_skip_4x)
        features_skip_4x_reduced = self.bn4x(features_skip_4x_reduced)
        features_skip_4x_reduced = self.relu(features_skip_4x_reduced)

        features_skip_2x_reduced = self.skip2x_to_reduced(features_skip_2x)
        features_skip_2x_reduced = self.bn2x(features_skip_2x_reduced)
        features_skip_2x_reduced = self.relu(features_skip_2x_reduced)
        

        # upsampling
        if self.if_up_bottleneck:
            features_bottleneck_4x = self.upconv_bottle1(features_bottleneck)
            features_bottleneck_4x = self.upconv_bottle2(features_bottleneck_4x)
        else: 
            features_bottleneck_4x = F.interpolate(
            features_skip_4x_reduced, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )

        # concat
        concat_features_4x = torch.cat((features_skip_4x_reduced, features_bottleneck_4x),dim=1)
        
        # refine 1
        features_4x = self.conv3x3_refine_4x(concat_features_4x)
        features_4x = self.bn_refine_4x(features_4x)
        features_4x = self.relu(features_4x)

        #upconv
        features_2x = self.upconv4x(features_4x)
        concat_features_2x = torch.cat((features_skip_2x_reduced, features_2x),dim=1)

        # refine2
        features_2x = self.conv3x3_refine_2x(concat_features_2x)
        features_2x = self.bn_refine_2x(features_2x)
        features_2x = self.relu(features_2x)

        # predict
        predictions_2x = self.features_to_predictions(features_2x)
        return predictions_2x, features_2x



class DecoderDeeplabV3pConv2xSkipFinal(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_2x_ch, num_out_ch, feature_out_ch):
        super(DecoderDeeplabV3pConv2xSkipFinal, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following
        self.features_to_predictions = torch.nn.Conv2d(feature_out_ch, num_out_ch, kernel_size=1, stride=1)
        

        self.skip2x_to_reduced = torch.nn.Conv2d(skip_2x_ch, 24, kernel_size=1, stride=1, bias=False)
        self.bn2x = torch.nn.BatchNorm2d(24)
        self.relu = torch.nn.ReLU()
        self.conv3x3_refine_2x = torch.nn.Conv2d(24 + bottleneck_ch, feature_out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_refine_2x = torch.nn.BatchNorm2d(feature_out_ch)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, features_2x, features_skip_2x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        
        # skip
        features_skip_2x_reduced = self.skip2x_to_reduced(features_skip_2x)
        features_skip_2x_reduced = self.bn2x(features_skip_2x_reduced)
        features_skip_2x_reduced = self.relu(features_skip_2x_reduced)

        #concate
        concat_features_2x = torch.cat((features_skip_2x_reduced, features_2x),dim=1)

        # refine2
        features_2x = self.conv3x3_refine_2x(concat_features_2x)
        features_2x = self.bn_refine_2x(features_2x)
        features_2x = self.relu(features_2x)

        # predict
        predictions_2x = self.features_to_predictions(features_2x)
        return predictions_2x, features_2x