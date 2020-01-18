import math
import torch
from torch import nn
from tqdm import tqdm
from configuration import cfg, device
from models.transformer.optim import TransformerScheduler
from readers.data_provider import train_iter, val_iter, src_val_file_address, tgt_val_file_address
from utils.optimizers import get_a_new_optimizer
from models.transformer.model import Transformer
from utils.init_nn import weight_init
from utils.evaluation import evaluate_transformer as evaluate


def main_transformer():
    from readers.data_provider import SRC, TGT
    model = Transformer(SRC, TGT).to(device)
    model.init_model_params()
    torch.save({'model': model, 'field_src': SRC, 'field_tgt': TGT}, cfg.checkpoint_name)
    # optimizer, scheduler = get_a_new_optimizer(cfg.init_optim, cfg.init_learning_rate, model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    model_size = int(cfg.transformer_d_model)
    factor, warmup = int(cfg.transformer_opt_factor), int(cfg.transformer_opt_warmup)
    scheduler = TransformerScheduler(model_size, factor, warmup, optimizer)
    size_train = len([_ for _ in train_iter])
    val_indices = [int(size_train * x / float(cfg.val_slices)) for x in range(1, int(cfg.val_slices))]
    if bool(cfg.debug_mode):
        evaluate(val_iter, TGT, model, src_val_file_address, tgt_val_file_address, "INIT")
    best_val_score = 0.0
    for epoch in range(int(cfg.n_epochs)):
        all_loss = 0.0
        batch_count = 0.0
        all_perp = 0.0
        all_tokens_count = 0.0
        ds = tqdm(train_iter, total=size_train)
        for ind, instance in enumerate(ds):
            optimizer.zero_grad()
            if instance.src[0].size(0) < 2:
                continue
            _, lss, decoded_length, n_tokens = model(instance.src, instance.trg)
            itm = lss.item()
            all_loss += itm
            all_tokens_count += float(n_tokens.item())
            all_perp += math.exp(itm / max(n_tokens, 1.0))
            batch_count += 1.0
            lss /= max(decoded_length, 1)
            lss.backward()
            #if bool(cfg.grad_clip):
            #    nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
            # scheduler.step()
            optimizer.step()
            scheduler.step()
            current_perp = all_perp / batch_count
            if current_perp < 1500:
                ds.set_description("Epoch: {}, Average Loss: {:.2f}, Average Perplexity: {:.2f}".format(
                    epoch, all_loss / all_tokens_count, current_perp))
            else:
                ds.set_description("Epoch: {}, Average Loss: {:.2f}".format(epoch, all_loss / all_tokens_count))
            if ind in val_indices:
                val_l, val_bleu = evaluate(val_iter, TGT, model, src_val_file_address, tgt_val_file_address, str(epoch))
                if val_bleu > best_val_score:
                    torch.save({'model': model, 'field_src': SRC, 'field_tgt': TGT}, cfg.checkpoint_name)
                    best_val_score = val_bleu

    if best_val_score > 0.0:
        print("Loading the best validated model with validation bleu score of {:.3f}".format(best_val_score))
        saved_obj = torch.load(cfg.checkpoint_name, map_location=lambda storage, loc: storage)
        model = saved_obj['model'].to(device)
        # it might not correctly overwrite the vocabulary objects
        SRC = saved_obj['field_src']
        TGT = saved_obj['field_tgt']
    evaluate(val_iter, TGT, model, src_val_file_address, tgt_val_file_address, "LAST")


if __name__ == "__main__":
    main_transformer()

