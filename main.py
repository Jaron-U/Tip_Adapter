import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import clip

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    cfg = yaml.load(open("custom_dataset.yaml", 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('cache', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # prepare dataset
    random.seed(1)
    torch.manual_seed(1)


    
    pass

if __name__ == '__main__':
    main()