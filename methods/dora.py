# dora.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dino import DINO

class Dora(DINO):
    def __init__(self, student_network, teacher_network):
        super().__init__(student_network, teacher_network)

    def training_step(self, batch):
        images, seg_maps = batch
        assert isinstance(images, list) and isinstance(seg_maps, list), \
            "Batch must contain a list of image tensors and a list of segmentation map tensors."

        for img, seg_map in zip(images, seg_maps):
            chosen_masks = self._select_random_masks(seg_map, num_masks=5)

            for mask in chosen_masks:
                masked_img = img * mask
                student_output = self.student_forward(masked_img)

    def _select_random_masks(self, seg_map, num_masks=5):
        """Select random segmentation masks from the segmentation map."""
        unique_labels = seg_map.unique()
        chosen_labels = random.sample(list(unique_labels), k=num_masks)
        masks = [(seg_map == label).float() for label in chosen_labels]
        return masks


    def student_forward(self, image):
        """Forward pass for the student network, as defined in DINO."""
        return self.student(image)

