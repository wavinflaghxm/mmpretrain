import torch
import torch.nn as nn
from torch.nn import functional as F

from mmengine.dist import get_dist_info
from mmpretrain.registry import MODELS
from mmpretrain.models import weight_reduce_loss
from mmpretrain.evaluation.metrics import Accuracy

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_feature(image_feats,
                    text_feats,
                    local_loss=False,
                    gather_with_grad=False,
                    rank=0,
                    world_size=1,
                    use_horovod=False):
    assert has_distributed, 'torch.distributed did not import correctly, ' \
                            'please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_feats = hvd.allgather(image_feats)
            all_text_feats = hvd.allgather(text_feats)
        else:
            with torch.no_grad():
                all_image_feats = hvd.allgather(image_feats)
                all_text_feats = hvd.allgather(text_feats)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_feats = list(all_image_feats.chunk(world_size, dim=0))
                gathered_text_feats = list(all_text_feats.chunk(world_size, dim=0))
                gathered_image_feats[rank] = image_feats
                gathered_text_feats[rank] = text_feats
                all_image_feats = torch.cat(gathered_image_feats, dim=0)
                all_text_feats = torch.cat(gathered_text_feats, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_feats = torch.cat(torch.distributed.nn.all_gather(image_feats), dim=0)
            all_text_feats = torch.cat(torch.distributed.nn.all_gather(text_feats), dim=0)
        else:
            gathered_image_feats = [torch.zeros_like(image_feats) for _ in range(world_size)]
            gathered_text_feats = [torch.zeros_like(text_feats) for _ in range(world_size)]
            dist.all_gather(gathered_image_feats, image_feats)
            dist.all_gather(gathered_text_feats, text_feats)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_feats[rank] = image_feats
                gathered_text_feats[rank] = text_feats
            all_image_feats = torch.cat(gathered_image_feats, dim=0)
            all_text_feats = torch.cat(gathered_text_feats, dim=0)

    return all_image_feats, all_text_feats


@MODELS.register_module()
class CLIPLoss(nn.Module):

    def __init__(self,
                 local_loss: bool = False,
                 gather_with_grad: bool = False,
                 cache_labels: bool = False,
                 use_horovod: bool = False,
                 reduction: str = 'sum',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.use_horovod = use_horovod
        self.reduction = reduction
        self.loss_weight = loss_weight

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_feats, text_feats, logit_scale):
        if self.world_size > 1:
            all_image_feats, all_text_feats = gather_feature(
                image_feats, text_feats,
                self.local_loss, self.gather_with_grad, self.rank, 
                self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_feats @ all_text_feats.T
                logits_per_text = logit_scale * text_feats @ all_image_feats.T
            else:
                logits_per_image = logit_scale * all_image_feats @ all_text_feats.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_feats @ text_feats.T
            logits_per_text = logit_scale * text_feats @ image_feats.T
        
        return logits_per_image, logits_per_text

    def forward(self, 
                image_feats, 
                text_feats, 
                logit_scale,
                weight=None,
                reduction_override=None,
                avg_factor=None,
                cal_acc=False):
        device = image_feats.device
        logits_per_image, logits_per_text = self.get_logits(image_feats, text_feats, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        loss = (F.cross_entropy(logits_per_image, labels) + 
                F.cross_entropy(logits_per_text, labels)) / 2

        # apply weights and do the reduction
        if weight is not None:
            assert weight.dim() == 1
            weight = weight.float()
            if loss.dim() > 1:
                weight = weight.reshape(-1, 1)
        
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        # compute accuracy
        acc = Accuracy.calculate(logits_per_image, labels) if cal_acc else None
        
        return loss, acc
