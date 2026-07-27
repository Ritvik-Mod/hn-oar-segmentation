"""
Custom nnU-Net v2 trainer: 250 epochs, mirror augmentation DISABLED.

Rationale: nnU-Net's default left-right mirror augmentation (and test-time
mirroring) makes left/right-distinct structures interchangeable, which caused
the model to confuse PAROTID_L vs PAROTID_R on a subset of cases. Disabling
mirroring lets the model learn a consistent, correct side assignment.

NOTE on __init__: we must replicate the base trainer's exact signature (not
*args/**kwargs). nnUNetTrainer.__init__ records its init kwargs by iterating over
signature(self.__init__).parameters and reading them from locals(); a *args
signature yields the names 'args'/'kwargs', which aren't in that frame -> KeyError.

Install: copy this file into the nnU-Net package trainer variants directory, e.g.
  NNUNET_DIR=$(python3 -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
  cp nnUNetTrainer_250epochs_noMirror.py $NNUNET_DIR/training/nnUNetTrainer/variants/
Then train with:  nnUNetv2_train <id> 3d_fullres 0 -tr nnUNetTrainer_250epochs_noMirror
And predict with: nnUNetv2_predict ... --disable_tta
"""
import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainerNoMirroring,
)


class nnUNetTrainer_250epochs_noMirror(nnUNetTrainerNoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
