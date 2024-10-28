# dora.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from seguidora.methods.dino import DINO
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import update_momentum

class Seguidora(DINO):
    def __init__(self, student_network, teacher_network):
        super().__init__(student_network, teacher_network)

    def training_step(self, batch):
        """Training step for the Seguidora model."""
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.996,
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.backbone, m=momentum)
        update_momentum(self.student_projection_head, self.projection_head, m=momentum)

        views, seg_maps = batch
        global_views = torch.cat(views[:2])
        local_views = torch.cat(views[2:])

        teacher_features = self.forward(global_views).flatten(start_dim=1)
        teacher_projections = self.projection_head(teacher_features)

        loss = 0
        student_projections = []
        for _ in range(5):
            chosen_masks = self._select_random_masks(seg_maps[2:])
            local_views_masked = local_views * chosen_masks
            student_projections = torch.cat(
                [self.forward_student(global_views), self.forward_student(local_views_masked)]
            )
            loss += self.criterion(
                teacher_out=teacher_projections.chunk(2),
                student_out=student_projections.chunk(len(views)),
                epoch=self.current_epoch,
            )

        self.log_dict(
            {"train/loss": loss, "train/ema_momentum": momentum},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(views),
        )
        return loss


    def _select_random_masks(self, seg_maps):
        """Selects a random mask from each element in the batch of segmentation masks."""
        selected_masks = []
        for seg_map in seg_maps:
            num_objects = seg_map.size(0)
            random_index = random.randint(0, num_objects - 1)
            selected_mask = seg_map[random_index]
            selected_masks.append(selected_mask)

        return torch.stack(selected_masks)
