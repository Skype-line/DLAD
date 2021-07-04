from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR

from mtl.datasets.dataset_miniscapes import DatasetMiniscapes
from mtl.models.model_deeplab_v3_plus import ModelDeepLabV3Plus
from mtl.models.model_deeplab_v3_plus_branch import ModelDeepLabV3PlusBranch
from mtl.models.model_deeplab_v3_plus_branch_attention import ModelDeepLabV3PlusBranchAttention
from mtl.models.model_deeplab_v3_plus_SA_SE import ModelDeepLabV3PlusSASE
from mtl.models.model_deeplab_v3_plus_SA_Gate import ModelDeepLabV3PlusGate
from mtl.models.model_deeplab_v3_plus_upconv import ModelDeepLabV3PlusUpConv
from mtl.models.model_deeplab_v3_plus_asppsase import ModelDeepLabV3PlusASPPSASE
from mtl.models.model_deeplab_v3_plus_upconv_skip2x_SE import ModelDeepLabV3PlusUpConvSkipSE
from mtl.models.model_deeplab_v3_plus_upconv_skip2x_RASE import ModelDeepLabV3PlusUpConvSkipRASE

def resolve_dataset_class(name):
    return {
        'miniscapes': DatasetMiniscapes,
    }[name]


def resolve_model_class(name):
    return {
        'deeplabv3p': ModelDeepLabV3Plus,
        'deeplabv3pbranch': ModelDeepLabV3PlusBranch,
        'deeplabv3pdistillation': ModelDeepLabV3PlusBranchAttention,
        'deeplabv3pSASE': ModelDeepLabV3PlusSASE,
        'deeplabv3pGate': ModelDeepLabV3PlusGate,
        'deeplabv3pUpConv': ModelDeepLabV3PlusUpConv,
        'deeplabv3pASPPSASE': ModelDeepLabV3PlusASPPSASE,
        'deeplabv3pUpConvSkipSE': ModelDeepLabV3PlusUpConvSkipSE,
        'deeplabv3pUpConvSkipRASE': ModelDeepLabV3PlusUpConvSkipRASE
    }[name]


def resolve_optimizer(cfg, params):
    if cfg.optimizer == 'sgd':
        return SGD(
            params,
            lr=cfg.optimizer_lr,
            momentum=cfg.optimizer_momentum,
            weight_decay=cfg.optimizer_weight_decay,
        )
    elif cfg.optimizer == 'adam':
        return Adam(
            params,
            lr=cfg.optimizer_lr,
            weight_decay=cfg.optimizer_weight_decay,
        )
    else:
        raise NotImplementedError


def resolve_lr_scheduler(cfg, optimizer):
    if cfg.lr_scheduler == 'poly':
        return LambdaLR(
            optimizer,
            lambda ep: max(1e-6, (1 - ep / cfg.num_epochs) ** cfg.lr_scheduler_power)
        )
    else:
        raise NotImplementedError
