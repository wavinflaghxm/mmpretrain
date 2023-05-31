import argparse

import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from mmpretrain.models import build_classifier

from projects.clip.datasets import (tokenize, OPENAI_IMAGENET_TEMPLATES,
                                    SIMPLE_IMAGENET_TEMPLATES)

MAX_N = 5000

TEMPLATE_TYPES = {
    'openai': OPENAI_IMAGENET_TEMPLATES,
    'simple': SIMPLE_IMAGENET_TEMPLATES,
}


def get_class_name(path: str):
    if 'objects365' in path.lower():
        names = []
        with open(path, 'r') as f:
            for line in f:
                tmp = line.strip().split(',')
                names.append(tmp[2])
    else:
        raise NotImplementedError
    
    return names


def dump_features(args):
    # bulid model
    cfg = Config.fromfile(args.config).model.backbone
    cfg.init_cfg.checkpoint = args.checkpoint
    model = build_classifier(cfg)
    model.init_weights()
    model.to(args.device)
    model.eval()

    names = get_class_name(args.class_name)
    num_names = len(names)

    templates = TEMPLATE_TYPES[args.template]
    num_templates = len(templates)

    use_format = isinstance(templates[0], str)
    text = [t.format(n) if use_format else t(n) 
            for n in names for t in templates]
    text = tokenize(text).reshape(num_names, num_templates,
                                  -1).to(args.device)

    with torch.no_grad():
        def text_split(text, n):
            for i in range(0, len(text), n):
                yield text[i: i + n]

        max_n = MAX_N // text.shape[1]
        text_feats = torch.cat([
            model.encode_text(x) for x in text_split(text, max_n)],
            dim=0)

    text_feats = F.normalize(text_feats.mean(dim=1), dim=-1)
    text_feats = text_feats.cpu().numpy()

    if args.out != '':
        print('saveing to', args.out)
        np.save(open(args.out, 'wb'), text_feats)


def main():
    parser = argparse.ArgumentParser(
        description='Dump CLIP text features.')
    parser.add_argument('config', help='model config file path.')
    parser.add_argument('checkpoint', help='checkpoint file.')
    parser.add_argument('--out', help='the file to output results.')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument(
        '--template', 
        choices=['openai', 'simple'],
        help='the template type is openai or simple.')
    parser.add_argument(
        '--class-name', 
        help='the file that holds the class name')
    args = parser.parse_args()
    dump_features(args)


if __name__ == '__main__':
    main()

# python projects/clip/tools/dump_text_features.py \
# projects/clip/config/vit-base-p16_pt-64xb64_in1k.py \
# data/pretrained/clip/CLIP-ViT-B-16-laion2B-s34B-b88K/pretrain.pth \
# --template 'openai' --class-name data/metadata/Objects365_names_fix.csv \
# --out work_dirs/cache/o365_clip_openai.npy