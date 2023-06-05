# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import torch
import mmengine


def convert_text(model_key, model_weight, state_dict, converted_names):
    new_key = 'text_backbone.' + model_key
    if 'token_embedding' in model_key:
        new_key = new_key.replace('token_embedding', 'token_embed')
        model_weight = model_weight
    if 'positional_embedding' in model_key:
        new_key = new_key.replace('positional_embedding', 'pos_embed')
    if 'text_projection' in model_key:
        new_key = new_key.replace('text_backbone.text_projection', 'text_projection')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_visual(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('visual.', 'vision_backbone.')
    if 'transformer.' in model_key:
        new_key = new_key.replace('transformer.', '')
    if 'resblocks' in model_key:
        new_key = new_key.replace('resblocks', 'layers')
        new_key = new_key.replace('ln_', 'ln')
        new_key = new_key.replace('in_proj_weight', 'qkv.weight')
        new_key = new_key.replace('in_proj_bias', 'qkv.bias')
        new_key = new_key.replace('out_proj.weight', 'proj.weight')
        new_key = new_key.replace('out_proj.bias', 'proj.bias')
        new_key = new_key.replace('mlp.c_fc', 'ffn.layers.0.0')
        new_key = new_key.replace('mlp.c_proj', 'ffn.layers.1')
    if 'ln_pre' in model_key:
        new_key = new_key.replace('ln_pre', 'pre_norm')
    if 'class_embedding' in model_key:
        new_key = new_key.replace('class_embedding', 'cls_token')
        model_weight = model_weight[None, None, :]
    if 'positional_embedding' in model_key:
        new_key = new_key.replace('positional_embedding', 'pos_embed')
        model_weight = model_weight[None, ...]
    if 'ln_post' in model_key:
        new_key = new_key.replace('ln_post', 'ln1')
    if 'visual.conv1' in model_key:
        new_key = new_key.replace('conv1', 'patch_embed.projection')
    if 'visual.proj' in model_key:
        new_key = new_key.replace('vision_backbone.proj', 'vision_projection')
    
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert(src, dst):
    """Convert keys in pycls pretrained RegNet models to mmdet style."""
    # load model
    ori_model = torch.load(src)
    # convert to pytorch style
    state_dict = OrderedDict()
    converted_names = set()
    for key, weight in ori_model.items():
        if 'logit_scale' in key:
            state_dict[key] = weight
            converted_names.add(key)
            print(f'Convert {key} to {key}')
        elif 'visual' in key:
            convert_visual(key, weight, state_dict, converted_names)
        else:
            convert_text(key, weight, state_dict, converted_names)

    # check if all layers are converted
    for key in ori_model.keys():
        if key not in converted_names:
            print(f'not converted: {key}')
    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict

    mmengine.mkdir_or_exist(osp.dirname(dst))
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained open clip models to mmdet style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
