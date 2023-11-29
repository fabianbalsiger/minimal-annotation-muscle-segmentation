"""Custom nnU-Net trainer allowing to ignore unsegmented image slices in a volume when computing the loss."""
from __future__ import annotations

from pathlib import Path

import batchgenerators.utilities.file_and_folder_operations as ffops
import nnunet.training.network_training.nnUNetTrainerV2 as trainerV2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss


class DC_and_CE_loss_improved(DC_and_CE_loss):
    """Wrapper of the DC_and_CE_loss that does return 0 instead of NaN when the target only consists of the ignored label.

    The problem of the NaN arises due to a division by zero at the line 'ce_loss = ce_loss.sum() / mask.sum()'.
    In such a case, the mask.sum() is simply zero as the target consists only of ignored labels.
    """

    def forward(self, net_output, target):
        """Calculate the Dice and cross-entropy loss."""
        if self.ignore_label is not None and target.unique().size(dim=0) == 1:
            if self.ignore_label == target.unique():
                return 0
        return super().forward(net_output, target)


class nnUNetTrainerV2_MUSEGAI(trainerV2.nnUNetTrainerV2):
    """Custom nnUNetTrainer that ignores the background allowing to train a full 3-D volume segmentation with sparse annotations.

    See Also: Çiçek et al., 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, MICCAI 2016, https://arxiv.org/abs/1606.06650
    """

    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
    ):
        """Initialize the nnU-Net trainer for muscle segmentation."""
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
        )
        if Path(self.plans_file).is_file():
            # during training, we do have access to the plans_file
            plans = ffops.load_pickle(self.plans_file)
        else:
            # during inference, we might not have access to the plans_file
            plans = {"num_classes": None}

        # ignore the background label for training with every n-th image slice in 3-D volumes
        self.loss = DC_and_CE_loss_improved(
            {"batch_dice": self.batch_dice, "smooth": 1e-5, "do_bg": False},
            {},
            ignore_label=plans["num_classes"],
        )
