import os

from torch.hub import download_url_to_file
import torch.nn as nn
import torch
from diffusers import UNet2DConditionModel
from diffusers.configuration_utils import FrozenDict


def patch_transvae_sd(model, state_dict):
    return {'model.' + k: v for k, v in state_dict.items()}


def module_dtype(self):
    return next(self.parameters()).dtype


def module_device(self):
    return next(self.parameters()).device


def conv_add_channels(new_c: int, conv: nn.Conv2d, prepend=False):
    
    new_conv = nn.Conv2d(new_c + conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias is not None)
    
    sd = conv.state_dict()
    ks = conv.kernel_size[0]
    if prepend:
        sd['weight'] = torch.cat([torch.zeros((conv.out_channels, new_c, ks, ks)), sd['weight']], dim=1)
    else:
        sd['weight'] = torch.cat([sd['weight'], torch.zeros((conv.out_channels, new_c, ks, ks))], dim=1)

    new_conv.load_state_dict(sd, strict=True)
    new_conv.to(device=module_device(conv), dtype=module_dtype(conv))

    return new_conv


def update_net_config(net: UNet2DConditionModel, key: str, value):
    new_config = dict(net.config)
    new_config[key] = value
    net._internal_dict = FrozenDict(new_config)


def patch_unet_convin(unet: UNet2DConditionModel, target_in_channels, prepend=False):
    '''
    add new channels to unet.conv_in, weights init to zeros
    '''

    new_added_conv_channels = target_in_channels - unet.config.in_channels
    if new_added_conv_channels < 1:
        return
    new_conv = conv_add_channels(new_added_conv_channels, unet.conv_in, prepend=prepend)
    del unet.conv_in
    unet.conv_in = new_conv
    update_net_config(unet, "in_channels", new_conv.in_channels)



def download_model(url, local_path):
    if os.path.exists(local_path):
        return local_path

    temp_path = local_path + '.tmp'
    download_url_to_file(url=url, dst=temp_path)
    os.rename(temp_path, local_path)
    return local_path


def load_frozen_patcher(filename, state_dict, strength):
    patch_dict = {}
    for k, w in state_dict.items():
        model_key, patch_type, weight_index = k.split('::')
        if model_key not in patch_dict:
            patch_dict[model_key] = {}
        if patch_type not in patch_dict[model_key]:
            patch_dict[model_key][patch_type] = [None] * 16
        patch_dict[model_key][patch_type][int(weight_index)] = w

    patch_flat = {}
    for model_key, v in patch_dict.items():
        for patch_type, weight_list in v.items():
            patch_flat[model_key] = (patch_type, weight_list)

    add_patches(filename=filename, patches=patch_flat, strength_patch=float(strength), strength_model=1.0)
    return


def add_patches(self, *, filename, patches, strength_patch=1.0, strength_model=1.0, online_mode=False):
    lora_identifier = (filename, strength_patch, strength_model, online_mode)
    this_patches = {}

    p = set()
    model_keys = set(k for k, _ in self.model.named_parameters())

    for k in patches:
        offset = None
        function = None

        if isinstance(k, str):
            key = k
        else:
            offset = k[1]
            key = k[0]
            if len(k) > 2:
                function = k[2]

        if key in model_keys:
            p.add(k)
            current_patches = this_patches.get(key, [])
            current_patches.append([strength_patch, patches[k], strength_model, offset, function])
            this_patches[key] = current_patches

    self.lora_patches[lora_identifier] = this_patches
    return p


# class LoraLoader:
#     def __init__(self, model):
#         self.model = model
#         self.backup = {}
#         self.online_backup = []
#         self.loaded_hash = str([])

#     @torch.inference_mode()
#     def refresh(self, lora_patches, offload_device=torch.device('cpu'), force_refresh=False):
#         hashes = str(list(lora_patches.keys()))

#         if hashes == self.loaded_hash and not force_refresh:
#             return

#         # Merge Patches

#         all_patches = {}

#         for (_, _, _, online_mode), patches in lora_patches.items():
#             for key, current_patches in patches.items():
#                 all_patches[(key, online_mode)] = all_patches.get((key, online_mode), []) + current_patches

#         # Initialize

#         memory_management.signal_empty_cache = True

#         parameter_devices = get_parameter_devices(self.model)

#         # Restore

#         for m in set(self.online_backup):
#             del m.forge_online_loras

#         self.online_backup = []

#         for k, w in self.backup.items():
#             if not isinstance(w, torch.nn.Parameter):
#                 # In very few cases
#                 w = torch.nn.Parameter(w, requires_grad=False)

#             utils.set_attr_raw(self.model, k, w)

#         self.backup = {}

#         set_parameter_devices(self.model, parameter_devices=parameter_devices)

#         # Patch

#         for (key, online_mode), current_patches in all_patches.items():
#             try:
#                 parent_layer, child_key, weight = utils.get_attr_with_parent(self.model, key)
#                 assert isinstance(weight, torch.nn.Parameter)
#             except:
#                 raise ValueError(f"Wrong LoRA Key: {key}")

#             if online_mode:
#                 if not hasattr(parent_layer, 'forge_online_loras'):
#                     parent_layer.forge_online_loras = {}

#                 parent_layer.forge_online_loras[child_key] = current_patches
#                 self.online_backup.append(parent_layer)
#                 continue

#             if key not in self.backup:
#                 self.backup[key] = weight.to(device=offload_device)

#             bnb_layer = None

#             if hasattr(weight, 'bnb_quantized') and operations.bnb_avaliable:
#                 bnb_layer = parent_layer
#                 from backend.operations_bnb import functional_dequantize_4bit
#                 weight = functional_dequantize_4bit(weight)

#             gguf_cls = getattr(weight, 'gguf_cls', None)
#             gguf_parameter = None

#             if gguf_cls is not None:
#                 gguf_parameter = weight
#                 from backend.operations_gguf import dequantize_tensor
#                 weight = dequantize_tensor(weight)

#             try:
#                 weight = merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)
#             except:
#                 print('Patching LoRA weights out of memory. Retrying by offloading models.')
#                 set_parameter_devices(self.model, parameter_devices={k: offload_device for k in parameter_devices.keys()})
#                 memory_management.soft_empty_cache()
#                 weight = merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)

#             if bnb_layer is not None:
#                 bnb_layer.reload_weight(weight)
#                 continue

#             if gguf_cls is not None:
#                 gguf_cls.quantize_pytorch(weight, gguf_parameter)
#                 continue

#             utils.set_attr_raw(self.model, key, torch.nn.Parameter(weight, requires_grad=False))

#         # End

#         set_parameter_devices(self.model, parameter_devices=parameter_devices)
#         self.loaded_hash = hashes
#         return
