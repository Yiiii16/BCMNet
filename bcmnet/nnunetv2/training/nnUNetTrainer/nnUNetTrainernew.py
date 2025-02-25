from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.bcmnet import get_bcmnet_3d_from_plans


class nnUNetTrainernew(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 3:
            model = get_bcmnet_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("UMambaRes: {}".format(model))

        return model
