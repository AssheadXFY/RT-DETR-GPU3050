"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision

import copy

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


@register()
class RTDETRCriterionv2(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        boxes_weight_format=None,
        share_matched_indices=False,
        deep_supervision=False,
        kd_temperature=1.0):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            boxes_weight_format: format for boxes weight (iou, )
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.deep_supervision = deep_supervision
        self.kd_temperature = kd_temperature

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
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

    def loss_labels_kd(self, outputs, dn_outputs, targets, indices, dn_meta):
        """Self-distillation: regular queries learn class distributions
           from denoising query predictions (soft labels).
           Denoising queries have known GT labels + box noise; after decoder
           they produce high-quality soft targets retaining model uncertainty.
        """
        device = outputs['pred_logits'].device
        pred_logits = outputs['pred_logits']          # [bs, nq, num_classes]
        dn_logits = dn_outputs['pred_logits']         # [bs, ndn, num_classes]
        dn_positive_idx = dn_meta['dn_positive_idx']
        dn_num_group = dn_meta['dn_num_group']
        bs = len(targets)

        kd_loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(bs):
            num_gt = len(targets[b]['labels'])
            if num_gt == 0 or len(indices[b][0]) == 0:
                continue

            reg_q_idx, reg_gt_idx = indices[b]
            dn_q_idx = dn_positive_idx[b]  # [num_gt * dn_num_group]

            if len(dn_q_idx) == 0:
                continue

            dn_per_gt = dn_q_idx.reshape(num_gt, dn_num_group)  # [num_gt, dn_num_group]

            for q, gt in zip(reg_q_idx, reg_gt_idx):
                dn_idxs = dn_per_gt[gt]

                with torch.no_grad():
                    dn_avg = dn_logits[b, dn_idxs].mean(dim=0)

                kd_loss += F.kl_div(
                    F.log_softmax(pred_logits[b, q] / self.kd_temperature, dim=-1),
                    F.softmax(dn_avg / self.kd_temperature, dim=-1),
                    reduction='sum',
                )
                count += 1

        if count > 0:
            kd_loss = kd_loss / count

        return {'loss_kd': kd_loss}

    def loss_group_consistency(self, main_outputs_list, indices_list, targets):
        """Cross-group consistency for query perturbation.

        For each GT matched by K >= 2 groups, the group with highest IoU
        to GT serves as anchor (stop_grad). Other groups' predictions
        are pulled towards the anchor via KL divergence (classification)
        and L1 loss (box regression).
        """
        K = len(main_outputs_list)
        assert K >= 2
        device = main_outputs_list[0]['pred_logits'].device
        bs = len(targets)

        loss_kl = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)
        count = 0

        for b in range(bs):
            num_gt = len(targets[b]['labels'])
            if num_gt == 0:
                continue
            gt_boxes_xyxy = box_cxcywh_to_xyxy(targets[b]['boxes'])

            for tgt_j in range(num_gt):
                gt_box = gt_boxes_xyxy[tgt_j]
                group_preds = []

                for g in range(K):
                    src_idx_g, tgt_idx_g = indices_list[g][b]
                    match = (tgt_idx_g == tgt_j)
                    if match.sum() == 0:
                        continue
                    q = src_idx_g[match][0].item()
                    logits_g = main_outputs_list[g]['pred_logits'][b, q]
                    box_g = main_outputs_list[g]['pred_boxes'][b, q]

                    iou, _ = box_iou(
                        box_cxcywh_to_xyxy(box_g.unsqueeze(0)),
                        gt_box.unsqueeze(0))
                    iou_val = iou[0, 0].item()
                    group_preds.append((logits_g, box_g, iou_val))

                if len(group_preds) < 2:
                    continue

                # anchor = highest IoU
                group_preds.sort(key=lambda x: x[2], reverse=True)
                anchor_logits, anchor_boxes, _ = group_preds[0]

                for logits_q, box_q, _ in group_preds[1:]:
                    with torch.no_grad():
                        anchor_dist = F.softmax(anchor_logits, dim=-1)
                    loss_kl += F.kl_div(
                        F.log_softmax(logits_q, dim=-1),
                        anchor_dist,
                        reduction='sum')
                    loss_box += F.l1_loss(box_q, anchor_boxes.detach(), reduction='sum')
                    count += 1

        if count > 0:
            loss_kl = loss_kl / count
            loss_box = loss_box / count

        return {'loss_consist_cls': loss_kl, 'loss_consist_box': loss_box}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }
        if loss == 'kd' or loss == 'consistency':
            return {}  # handled separately in forward
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # query perturbation: per-group loss averaging
        num_groups = outputs.get('num_query_groups', 1)
        if num_groups > 1 and 'query_group_aux' in outputs:
            losses = {}
            # per-group main loss (last layer)
            main_outputs_list = outputs['query_group_main']
            matched_indices_list = []
            for g in range(num_groups):
                g_out = main_outputs_list[g]
                matched = self.matcher(g_out, targets)
                indices_g = matched['indices']
                matched_indices_list.append(indices_g)
                for loss in self.losses:
                    if loss == 'consistency':
                        continue  # handled below
                    meta = self.get_loss_meta_info(loss, g_out, targets, indices_g)
                    l_dict = self.get_loss(loss, g_out, targets, indices_g, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_g{}'.format(g): v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # cross-group consistency: anchor-based alignment among K groups
            if 'consistency' in self.losses:
                consist_loss = self.loss_group_consistency(
                    main_outputs_list, matched_indices_list, targets)
                for k in consist_loss:
                    if k in self.weight_dict:
                        losses[k] = consist_loss[k] * self.weight_dict[k]

            # per-group aux loss
            for g in range(num_groups):
                group_aux = outputs['query_group_aux'][g]
                for i, aux_outputs in enumerate(group_aux):
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                    for loss in self.losses:
                        meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                        l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                        l_dict = {k + '_g{}_aux_{}'.format(g, i): v for k, v in l_dict.items()}
                        losses.update(l_dict)

            # enc / dn loss (unchanged)
            if 'enc_aux_outputs' in outputs:
                assert 'enc_meta' in outputs, ''
                class_agnostic = outputs['enc_meta']['class_agnostic']
                if class_agnostic:
                    orig_num_classes = self.num_classes
                    self.num_classes = 1
                    enc_targets = copy.deepcopy(targets)
                    for t in enc_targets:
                        t['labels'] = torch.zeros_like(t["labels"])
                else:
                    enc_targets = targets
                for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                    for loss in self.losses:
                        meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                        l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                        l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                        l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                if class_agnostic:
                    self.num_classes = orig_num_classes

            if 'dn_aux_outputs' in outputs:
                assert 'dn_meta' in outputs, ''
                indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
                dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
                for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                    for loss in self.losses:
                        meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                        l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                        l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

            # KD per-group: each group independently distills from dn queries
            if 'dn_meta' in outputs and 'dn_aux_outputs' in outputs and 'kd' in self.losses:
                for g in range(num_groups):
                    g_out = main_outputs_list[g]
                    matched = self.matcher(g_out, targets)
                    g_indices = matched['indices']
                    kd_loss = self.loss_labels_kd(
                        g_out, outputs['dn_aux_outputs'][-1],
                        targets, g_indices, outputs['dn_meta'])
                    kd_loss = {k + '_g{}'.format(g): kd_loss[k] * self.weight_dict.get('loss_kd', 1.0)
                               for k in kd_loss if k in self.weight_dict}
                    losses.update(kd_loss)

            return losses

        # === single-group path (baseline) ===
        # Retrieve the matching between the outputs of the last layer and the targets
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    if self.deep_supervision:
                        n_aux = len(outputs['aux_outputs'])
                        layer_weight = min(1.0, 0.3 * (i + 1))
                        l_dict = {k: v * layer_weight for k, v in l_dict.items()}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of encoder auxiliary losses. For rtdetr v2
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', ):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices
