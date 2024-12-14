import os
import random
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from custom_dataset import getDataset
from evaluator import Evaluator

def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, ft_evaluator: Evaluator, adapter=None):
    if cfg['search_hp'] == True:
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        best_f1 = 0
        best_beta, best_alpha = 0, 0

        for beta in tqdm(beta_list, desc="searing beta"):
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                ft_evaluator.reset()
                ft_evaluator.process(tip_logits.detach(), labels.detach())
                result = ft_evaluator.evaluate("tip_adapter", is_searching=True)

                if best_f1 < result['average_f1']:
                    best_beta = beta
                    best_alpha = alpha
                    best_f1 = result['average_f1']
                    tqdm.write(f"Processing beta={beta}, alpha={alpha}, New best F1: {best_f1:.3f}")
                    
        print(f"\n**** After searching, the best F1: {best_f1:.3f}")
        return best_beta, best_alpha

def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # tokenize the prompts ---- use template
            # classname = classname.replace('_', ' ')
            # texts = [t.format(classname) for t in template]
            # ----- directly use the label name -----
            texts = classname.replace('_', ' ')
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def build_cache_model(cfg, clip_model, train_loader_cache):
    if cfg['load_cache'] == False:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # data augmentaion for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print(f"Augment Epoch: {augment_idx} / {cfg['augment_epoch']}")
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt", weights_only=True)
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt", weights_only=True)
    
    return cache_keys, cache_values

def pre_load_features(cfg, split, clip_model, loader):
    if cfg['load_pre_feat'] == False:
        features, labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)
        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt", weights_only=True)
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt", weights_only=True)
    
    return features, labels

def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, 
                    clip_weights, evaluator:Evaluator, ft_evaluator:Evaluator):
    # zero-shot clip
    clip_logits = 100. * test_features @ clip_weights
    evaluator.reset()
    evaluator.process(clip_logits, test_labels)
    result = evaluator.evaluate("clip_adapter", is_searching=False)
    print(f"\n**** Zero-shot CLIP Average | Precision: {result['average_precision']:.3f} |"
          f"Recall: {result['average_recall']:.3f} | F1: {result['average_f1']:.3f}")

    # Tip_adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    tip_logits = clip_logits + cache_logits * alpha
    ft_evaluator.reset()
    ft_evaluator.process(tip_logits, test_labels)
    ft_result = ft_evaluator.evaluate("tip_adapter", is_searching=False)
    print(f"\n**** Tip-Adapter Average | Precision: {ft_result['average_precision']:.3f} |" 
          f"Recall: {ft_result['average_recall']:.3f} | F1: {ft_result['average_f1']:.3f}")
    
    # Search Hyperparameters
    # _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, ft_evaluator)

def run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, 
                      clip_weights, clip_model, train_loader_F, ft_evaluator: Evaluator):
    # enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_f1, best_epoch = 0.0, 0

    for epoch in range(cfg['train_epoch']):
        adapter.train()
        loss_list = []
        f1_list = []
        
        for i, (images, target) in enumerate(tqdm(train_loader_F, desc=f"epoch {epoch} / {cfg['train_epoch']}")):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)
            loss_list.append(loss)

            ft_evaluator.reset()
            ft_evaluator.process(tip_logits.detach(), target.detach())
            result = ft_evaluator.evaluate("ft_tip_adapter", is_searching=True)
            f1_list.append(result["average_f1"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        tqdm.write(f"LR: {scheduler.get_last_lr()[0]:.6f}, average f1: {sum(f1_list)/len(f1_list):.3f}, "
                   f"average Loss: {sum(loss_list)/len(loss_list):.3f}")
    
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (0.83 - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha

        ft_evaluator.reset()
        ft_evaluator.process(tip_logits.detach(), test_labels.detach())
        result = ft_evaluator.evaluate("ft_tip_adapter", is_searching=True)
        
        if best_f1 < result['average_f1']:
            best_epoch = epoch
            best_f1 = result['average_f1']
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt", weights_only=True)
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_f1:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    _ = search_hp(cfg, affinity, cache_values, test_features, test_labels, clip_weights, ft_evaluator, adapter=adapter)


def main():
    cfg = yaml.load(open("custom_dataset.yaml", 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('cache', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(cfg['OUTPUT_DIR'], exist_ok=True)
    cfg['cache_dir'] = cache_dir

    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # prepare dataset
    random.seed(1)
    torch.manual_seed(1)

    train_dataset, test_dataset = getDataset(cfg['root_path'], preprocess, shot_num=cfg['shots'])

    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=8, shuffle=False)

    train_loader_cache = DataLoader(train_dataset, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = DataLoader(train_dataset, batch_size=256, num_workers=8, shuffle=True)

    evaluator = Evaluator(cfg, train_dataset.label2name)
    ft_evaluator = Evaluator(cfg, train_dataset.label2name, tip_adapter=True)

    print("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(train_dataset.classnames, train_dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

     # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, evaluator, ft_evaluator)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, 
                      clip_model, train_loader_F, ft_evaluator)

if __name__ == '__main__':
    main()