import os
import math
import torch
from torch import nn
from tqdm import tqdm
from configuration import cfg, device
from readers.data_provider import DataProvider
from utils.optimizers import get_a_new_optimizer
from models.sts.model import STS
from models.transformer.model import Transformer
from models.aspect_extractor.model import AspectAugmentedTransformer
from models.aspect_extractor.syntax_infused_model import SyntaxInfusedTransformer
from models.transformer.optim import TransformerScheduler
from utils.init_nn import weight_init
from utils.evaluation import evaluate
from timeit import default_timer as timer


def print_running_time(t):
    day = t // (24 * 3600)
    t = t % (24 * 3600)
    hour = t // 3600
    t %= 3600
    minutes = t // 60
    t %= 60
    seconds = t
    print("Total training execution time: {:.2f} days, {:.2f} hrs, {:.2f} mins, {:.2f} secs".format(day, hour, minutes, seconds))


def create_sts_model(SRC, TGT):
    model = STS(SRC, TGT).to(device)
    model.apply(weight_init)
    optimizer, scheduler = get_a_new_optimizer(cfg.init_optim, cfg.init_learning_rate, model.parameters())
    return model, optimizer, scheduler, bool(cfg.grad_clip), True


def create_transformer_model(model_type, SRC, TGT):
    model = model_type(SRC, TGT).to(device)
    model.init_model_params()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, betas=(0.9, 0.98), eps=1e-9)
    model_size = int(cfg.transformer_d_model)
    factor, warmup = int(cfg.transformer_opt_factor), int(cfg.transformer_opt_warmup)
    scheduler = TransformerScheduler(model_size, factor, warmup, optimizer)
    return model, optimizer, scheduler, False, False


def main(model_name):
    dp = DataProvider()
    if model_name == "sts":
        model, optimizer, scheduler, grad_clip, step_only_at_eval = create_sts_model(dp.SRC, dp.TGT)
    elif model_name == "transformer":
        model, optimizer, scheduler, grad_clip, step_only_at_eval = create_transformer_model(Transformer, dp.SRC, dp.TGT)
    elif model_name == "aspect_augmented_transformer":
        model, optimizer, scheduler, grad_clip, step_only_at_eval = create_transformer_model(AspectAugmentedTransformer, dp.SRC, dp.TGT)
    elif model_name == "syntax_infused_transformer":
        model, optimizer, scheduler, grad_clip, step_only_at_eval = create_transformer_model(SyntaxInfusedTransformer, dp.SRC, dp.TGT)
    else:
        raise ValueError("Model name {} is not defined.".format(model_name))
    if not os.path.exists("../.checkpoints/"):
        os.mkdir("../.checkpoints/")
    training_evaluation_results = []
    torch.save({'model': model, 'field_src': dp.SRC, 'field_tgt': dp.TGT,
                'training_evaluation_results': training_evaluation_results}, "../.checkpoints/"+cfg.checkpoint_name)

    if bool(cfg.debug_mode):
        evaluate(dp.val_iter, dp, model, dp.processed_data.addresses.val.src, dp.processed_data.addresses.val.tgt, "INIT")
    best_val_score = 0.0
    assert cfg.update_freq > 0, "update_freq must be a non-negative integer"
    for epoch in range(int(cfg.n_epochs)):
        if epoch == int(cfg.init_epochs) and model_name == "sts":
            optimizer, scheduler = get_a_new_optimizer(cfg.optim, cfg.learning_rate, model.parameters())
        all_loss = 0.0
        batch_count = 0.0
        all_perp = 0.0
        all_tokens_count = 0.0
        if epoch < 2:
            # after the first iteration it does not need recalculation
            val_indices = [int(dp.size_train * x / float(cfg.val_slices)) for x in range(1, int(cfg.val_slices))]
        ds = tqdm(dp.train_iter, total=dp.size_train,  dynamic_ncols=True)
        optimizer.zero_grad()
        for ind, instance in enumerate(ds):
            if instance.src[0].size(0) < 2:
                continue
            pred, _, lss, decoded_length, n_tokens = model(instance.src, instance.trg, test_mode=False, **instance.data_args)
            itm = lss.item()
            all_loss += itm
            all_tokens_count += n_tokens
            all_perp += math.exp(itm / max(n_tokens, 1.0))
            batch_count += 1.0
            lss /= (max(decoded_length, 1) * cfg.update_freq)
            lss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
            if ind % cfg.update_freq == 0:
                """Implementation of gradient accumulation as suggested in https://arxiv.org/pdf/1806.00187.pdf"""
                optimizer.step()
                if not step_only_at_eval:
                    scheduler.step()
                optimizer.zero_grad()
            current_perp = all_perp / batch_count
            if current_perp < 1500:
                ds.set_description("Epoch: {}, Average Loss: {:.2f}, Average Perplexity: {:.2f}".format(
                    epoch, all_loss / all_tokens_count, current_perp))
            else:
                ds.set_description("Epoch: {}, Average Loss: {:.2f}".format(epoch, all_loss / all_tokens_count))
            if ind in val_indices:
                val_l, val_bleu = evaluate(dp.val_iter, dp, model, dp.processed_data.addresses.val.src, dp.processed_data.addresses.val.tgt, str(epoch))
                training_evaluation_results.append(val_bleu)
                if val_bleu > best_val_score:
                    torch.save({'model': model, 'field_src': dp.SRC, 'field_tgt': dp.TGT, 'training_evaluation_results': training_evaluation_results},
                               "../.checkpoints/"+cfg.checkpoint_name)
                    best_val_score = val_bleu
                if step_only_at_eval:
                    scheduler.step(val_bleu)
        dp.size_train = ind + 1

    if best_val_score > 0.0:
        print("Loading the best validated model with validation bleu score of {:.2f}".format(best_val_score))
        saved_obj = torch.load("../.checkpoints/"+cfg.checkpoint_name, map_location=lambda storage, loc: storage)
        model = saved_obj['model'].to(device)
        # it might not correctly overwrite the vocabulary objects
        SRC = saved_obj['field_src']
        TGT = saved_obj['field_tgt']
        dp.replace_fields(SRC, TGT)
    evaluate(dp.val_iter, dp, model, dp.processed_data.addresses.val.src, dp.processed_data.addresses.val.tgt, "LAST")


if __name__ == "__main__":
    start = timer()
    main(cfg.model_name)
    end = timer()
    print_running_time(end - start)

