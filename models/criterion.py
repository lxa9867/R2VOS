import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .segmentation import (dice_loss, sigmoid_focal_loss)

from einops import rearrange

class SetCriterion(nn.Module):
    """ This class computes the loss for ReferFormer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25, margin=2, use_cycle=False,
                 add_negative=False, neg_cls=False, quantitize_query=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_alpha = focal_alpha
        self.mask_out_stride = 4
        self.use_cycle = use_cycle
        self.add_negative = add_negative
        self.margin = margin
        self.neg_cls = neg_cls
        self.quantitize_query = quantitize_query

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] 
        _, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t q k -> b (t q) k')

        # judge the valid frames
        valid_indices = []
        valids = [target['valid'] for target in targets]
        for valid, (indice_i, indice_j) in zip(valids, indices): 
            valid_ind = valid.nonzero().flatten() 
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices) # NOTE: use valid indices
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) 
        if self.num_classes == 1: # binary referred
            target_classes[idx] = 0
        else:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            pass
        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']
        bs, nf, nq = src_boxes.shape[:3]
        src_boxes = src_boxes.transpose(1, 2)  

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]
        src_boxes = src_boxes.flatten(0, 1)  # [b*t, 4]
        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)  # [b*t, 4]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"] 
        src_masks = src_masks.transpose(1, 2)

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                              size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks) 

        # downsample ground truth masks with ratio mask_out_stride
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        src_masks = src_masks[src_idx] 
        # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False) 
        src_masks = src_masks.flatten(1) # [b, thw]

        target_masks = target_masks.flatten(1) # [b, thw]

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_fg_contra(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        temperature = 0.3
        mask_features = outputs['mask_features']  # btchw

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets],
                                                             size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(mask_features)

        # downsample ground truth masks with ratio mask_out_stride
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]

        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w
        target_masks = target_masks.flatten(1)  # [b, thw]
        target_masks_one_hot = F.one_hot(target_masks.long())  # b thw 2
        mask_features = rearrange(mask_features, 'b t c h w -> b (t h w) c')
        mask_average_feature = torch.einsum('bpd,bpi->bdi', mask_features, target_masks_one_hot.float())
        mask_areas = torch.sum(target_masks_one_hot, dim=1)
        mask_average_feature /= torch.maximum(mask_areas, torch.tensor(1)).unsqueeze(1)
        mask_average_feature = F.normalize(mask_average_feature, dim=1)
        foreground_discrimination_similarity = torch.einsum('bdi,bpd->bpi', mask_average_feature, mask_features)
        foreground_discrimination_similarity /= temperature
        losses = {
            'loss_fg_contra': F.cross_entropy(foreground_discrimination_similarity.flatten(0, 1),
                                              target_masks.flatten(0, 1).long())
        }
        return losses

    def loss_VQ(self, outputs, targets, indices, num_boxes):
        losses = {
            'loss_VQ': outputs['loss_VQ']
        }
        return losses

    def loss_loc(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "loc" in outputs

        # src_masks = outputs["pred_masks"]
        src_masks = outputs["loc"]
        src_masks = src_masks.transpose(0, 1)

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets],
                                                             size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks)

        # downsample ground truth masks with ratio mask_out_stride
        loc_stride = 32
        start = int(loc_stride // 2)
        im_h, im_w = target_masks.shape[-2:]

        target_masks = target_masks[:, :, start::loc_stride, start::loc_stride]
        assert target_masks.size(2) * loc_stride == im_h
        assert target_masks.size(3) * loc_stride == im_w
        import cv2
        # cv2.imwrite('lowres_tgt.png', np.concatenate((255 * target_masks[0, 0].cpu().numpy(), 255 * src_masks[0, 0].detach().cpu().numpy()), axis=0))

        # downsample gt to low resolusion
        # target_masks = interpolate(target_masks, size=src_masks.shape[-2:], mode="nearest")
        src_masks = src_masks.flatten(1)  # [b, thw]

        target_masks = target_masks.flatten(1)  # [b, thw]
        # print(src_masks.shape, target_masks.shape)

        losses = {
            "loss_loc": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses


    def RKdAngle(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

    def RkdDistance(self, student, teacher):
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss

    def ContrastiveLoss(self, neg_src, neg_tgt, pos_src, anchor, margin=1):
        neg_src_loss = F.pairwise_distance(anchor, neg_src, p=2)
        neg_tgt_loss = (margin - F.pairwise_distance(anchor, neg_tgt, p=2)).clamp(min=0)
        pos_src_loss = (margin - F.pairwise_distance(anchor, pos_src, p=2)).clamp(min=0)
        print('neg_src_loss', neg_src_loss, 'neg_tgt_loss', F.pairwise_distance(anchor, neg_tgt, p=2), 'pos_src_loss', F.pairwise_distance(anchor, pos_src, p=2))
        loss = torch.cat((neg_src_loss, neg_tgt_loss, pos_src_loss))
        return loss.mean()
        # with torch.no_grad():
        #     pos = pos
        # return F.triplet_margin_loss(anchor=anchor, positive=neg, negative=pos, margin=margin)

    def loss_cycle(self, outputs, targets, indices, num_boxes):
        src_sentence_embedding = outputs['pseudo_sentence_feature'].squeeze(0)  # BxC
        tgt_sentence_embedding = outputs['sentence_feature']

        # gather all
        src_sentence_embedding_list = [torch.zeros_like(src_sentence_embedding) for _ in range(dist.get_world_size())]
        tgt_sentence_embedding_list = [torch.zeros_like(tgt_sentence_embedding) for _ in range(dist.get_world_size())]
        dist.all_gather(src_sentence_embedding_list, src_sentence_embedding)
        dist.all_gather(tgt_sentence_embedding_list, tgt_sentence_embedding)

        if self.add_negative:
            neg_tgt_sentence_embedding = outputs['neg_sentence_feature']
            # negative_anchor = outputs['negative_anchor']
            neg_src_sentence_embedding = outputs['neg_pseudo_sentence_feature'].squeeze(0)  # BxC
            src_sentence_embedding_list[dist.get_rank()] = src_sentence_embedding
            tgt_sentence_embedding_list[dist.get_rank()] = tgt_sentence_embedding
            pos_src = torch.cat(src_sentence_embedding_list, dim=0)
            pos_tgt = torch.cat(tgt_sentence_embedding_list, dim=0)

            if self.neg_cls:
                pair_logits = outputs['pair_logits']
                pair_gt = outputs['pair_gt']

            # neg relation
            # gather all
            # neg_src_sentence_embedding_list = [torch.zeros_like(neg_src_sentence_embedding) for _ in
            #                                range(dist.get_world_size())]
            # neg_tgt_sentence_embedding_list = [torch.zeros_like(neg_tgt_sentence_embedding) for _ in
            #                                range(dist.get_world_size())]
            # dist.all_gather(neg_src_sentence_embedding_list, neg_src_sentence_embedding)
            # dist.all_gather(neg_tgt_sentence_embedding_list, neg_tgt_sentence_embedding)
            # neg_src_sentence_embedding_list[dist.get_rank()] = neg_src_sentence_embedding
            # neg_tgt_sentence_embedding_list[dist.get_rank()] = neg_tgt_sentence_embedding
            # neg_src = torch.cat(neg_src_sentence_embedding_list, dim=0)
            # neg_tgt = torch.cat(neg_tgt_sentence_embedding_list, dim=0)

            losses = {
                "loss_cycle_dist": self.RkdDistance(student=pos_src, teacher=pos_tgt),
                "loss_cycle_angle": self.RKdAngle(student=pos_src, teacher=pos_tgt),
                "loss_cycle_mse": F.pairwise_distance(src_sentence_embedding, tgt_sentence_embedding, p=2).mean(),
                "loss_cycle_contrastive":
                    (self.margin - F.pairwise_distance(neg_src_sentence_embedding, neg_tgt_sentence_embedding, p=2).mean()).clamp(min=0),
                # (self.margin - self.RkdDistance(student=neg_src, teacher=neg_tgt)).clamp(min=0)
                #self.ContrastiveLoss(neg_src_sentence_embedding, neg_tgt_sentence_embedding, src_sentence_embedding, src_sentence_embedding),
                # nn.functional.mse_loss(src_sentence_embedding, tgt_sentence_embedding),
            }
            # print('pos', F.pairwise_distance(src_sentence_embedding, tgt_sentence_embedding, p=2), 'neg',
            #       F.pairwise_distance(neg_src_sentence_embedding, neg_tgt_sentence_embedding, p=2), 'tgt',
            #       F.pairwise_distance(tgt_sentence_embedding, neg_tgt_sentence_embedding, p=2))
            if self.neg_cls:
                losses["loss_cycle_cls"] = F.binary_cross_entropy_with_logits(pair_logits, pair_gt)
        else:
            src_sentence_embedding_list[dist.get_rank()] = src_sentence_embedding
            tgt_sentence_embedding_list[dist.get_rank()] = tgt_sentence_embedding
            pos_src = torch.cat(src_sentence_embedding_list, dim=0)
            pos_tgt = torch.cat(tgt_sentence_embedding_list, dim=0)

            losses = {
                "loss_cycle_dist": self.RkdDistance(student=pos_src, teacher=pos_tgt),
                "loss_cycle_angle": self.RKdAngle(student=pos_src, teacher=pos_tgt),
                "loss_cycle_mse": F.pairwise_distance(src_sentence_embedding, tgt_sentence_embedding).mean(),
                #nn.functional.mse_loss(src_sentence_embedding, tgt_sentence_embedding),
            }
            # print('pos', F.pairwise_distance(src_sentence_embedding, tgt_sentence_embedding, p=2))
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'cycle': self.loss_cycle,
            'fg_contra': self.loss_fg_contra,
            'VQ': self.loss_VQ,
            # 'loc': self.loss_loc,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1) # [B, T] -> [B*T]
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'cycle' or loss == 'loc' or loss == 'fg_contra' or loss == 'VQ':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        # if self.add_negative:
        #     if dist.get_rank() == 0:
        #         for k, v in losses.items():
        #             if 'cycle' not in k:
        #                 v *= 0
        #     else:
        #         for k, v in losses.items():
        #             if 'cycle' not in k:
        #                 v *= dist.get_world_size() / (dist.get_world_size() - 1)

        return losses


