import sys, os, codecs, glob, io
import xml.etree.ElementTree as ET

import string
import math
import re
import numpy as np
import spacy
import torch
import yaml
import sacrebleu
from torch import nn, optim

import torch.nn.init as init
from pathlib import Path
from torchtext import data, datasets
# from NegativeSampling import LMWithNegativeSamples
from tqdm import tqdm
from collections import deque
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence















model = STS().to(device)
model.apply(weight_init)

torch.save({'model': model, 'field_src': SRC, 'field_tgt': TGT}, 'nmt.pt')
# saved_obj = torch.load('nmt.pt', map_location=lambda storage, loc: storage)
# model = saved_obj['model'].to(device)
# SRC = saved_obj['field_src']
# TGT = saved_obj['field_tgt']






optimizer, scheduler = get_a_new_optimizer(cfg.init_optim, cfg.init_learning_rate)

# scheduler = SGDRScheduler(optimizer, max_lr=float(cfg.lr_max), cycle_length=int(cfg.lr_cycle_length))











size_train = len([_ for _ in train_iter])
val_indices = [int(size_train * x / float(cfg.val_slices)) for x in range(1, int(cfg.val_slices))]

# past_losses = deque(maxlen=int(cfg.lr_decay_patience_steps))
if bool(cfg.debug_mode):
    validate("INIT")
best_val_score = 0.0
for epoch in range(int(cfg.n_epochs)):
    if epoch == int(cfg.init_epochs):
        optimizer, scheduler = get_a_new_optimizer(cfg.optim, cfg.learning_rate)
    all_loss = 0.0
    batch_count = 0.0
    all_perp = 0.0
    all_tokens_count = 0.0
    ds = tqdm(train_iter, total=size_train)
    for ind, instance in enumerate(ds):
        optimizer.zero_grad()
        if instance.src[0].size(0) < 2:
            continue
        pred, _, lss, decoded_length, n_tokens = model(instance.src, instance.trg)
        itm = lss.item()
        all_loss += itm
        all_tokens_count += n_tokens
        all_perp += math.exp(itm / max(n_tokens, 1.0))
        batch_count += 1.0
        lss /= max(decoded_length, 1)
        lss.backward()
        if bool(cfg.grad_clip):
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
        # scheduler.step()
        optimizer.step()
        current_perp = all_perp / batch_count
        if current_perp < 1500:
            ds.set_description("Epoch: {}, Average Loss: {:.2f}, Average Perplexity: {:.2f}".format(
                epoch, all_loss / all_tokens_count, current_perp))
        else:
            ds.set_description("Epoch: {}, Average Loss: {:.2f}".format(epoch, all_loss / all_tokens_count))
        if ind in val_indices:
            val_l, val_bleu = validate(epoch)
            if val_bleu > best_val_score:
                torch.save({'model': model, 'field_src': SRC, 'field_tgt': TGT}, 'nmt.pt')
                best_val_score = val_bleu
            scheduler.step(val_bleu)
            # past_losses.append(all_loss / all_tokens_count)
            # past_perp_diffs = sum([abs(past_losses[i + 1] - past_losses[i])
            #                       for i in range(len(past_losses) - 1)])
            # if past_perp_diffs < float(cfg.lr_decay_threshold) and \
            #        len(past_losses) == int(cfg.lr_decay_patience_steps):
            #    adjust_learning_rate(optimizer, float(cfg.lr_decay_factor))
            #    past_losses.clear()

if best_val_score > 0.0:
    print("Loading the best validated model with validation bleu score of {:.3f}".format(best_val_score))
    saved_obj = torch.load('nmt.pt', map_location=lambda storage, loc: storage)
    model = saved_obj['model'].to(device)
    SRC = saved_obj['field_src']
    TGT = saved_obj['field_tgt']
validate("LAST")