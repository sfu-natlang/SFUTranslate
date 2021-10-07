import string
import random
import torch
import os
import subprocess
from torch import nn
import torchtext
if torchtext.__version__.startswith('0.9') or torchtext.__version__.startswith('0.10'):
    from torchtext.legacy import data
else:
    from torchtext import data
import sacrebleu
from configuration import cfg
from readers.data_provider import DataProvider, src_tokenizer, tgt_detokenizer


def convert_to_sgml(file_name_to_be_converted, dataset_name, is_ref, src_lang, tgt_lang):
    """
    This function can be used to convert a text file (one sentence per line) to SGML format required for mteval-v14.pl
    Any source/reference dev/test file can be converted to SGML format using this function
    :param file_name_to_be_converted: the name of created evaluation file
    :param dataset_name: can be WMT19 or IWSLT17 or M30k or ...
    :param is_ref: a boolean flag showing whether :param eval_file_name: is from reference type or from source type
    :param src_lang: source language
    :param tgt_lang: target language
    """
    with open(file_name_to_be_converted + ".sgm", "w", encoding="utf-8") as out:
        out.write("""<{}set setid="{}" srclang="any"{}>\n<doc sysid="{}" docid="00000" genre="dataset" origlang="{}">\n<p>\n""".format(
            "ref" if is_ref else "src", dataset_name, " trglang=\"{}\"".format(tgt_lang) if is_ref else "", dataset_name, src_lang))
        with open(file_name_to_be_converted, "r", encoding="utf-8") as inp:
            for ind, line in enumerate(inp):
                if not len(line.strip()):
                    out.write("<seg id=\"{}\">{}</seg>\n".format(ind+1, "Empty"))
                else:
                    out.write("<seg id=\"{}\">{}</seg>\n".format(ind+1, line.strip()))
        out.write("""</p></doc>\n</{}set>""".format("ref" if is_ref else "src"))


def convert_target_batch_back(btch, TGT):
    """
    :param btch: seq_length, batch_size
    This function must look at attention scores for <unk> tokens and replace them with words from input if possible
    :return:
    """
    non_desired__ids = [TGT.vocab.stoi[cfg.pad_token], TGT.vocab.stoi[cfg.bos_token]]
    eos = TGT.vocab.stoi[cfg.eos_token]
    s_len, batch_size = btch.size()
    result = []
    for b in range(batch_size):
        tmp_res = []
        for w in range(s_len):
            itm = btch[w, b]
            if itm == eos:
                break
            if itm not in non_desired__ids:
                cvrted = TGT.vocab.itos[int(itm)]
                tmp_res.append(cvrted)
        result.append(" ".join(tmp_res))
    return result
    # return [" ".join([TGT.vocab.itos[int(btch[w, b])]
    #                  for w in range(s_len) if btch[w, b] not in non_desired__ids]) for b in range(batch_size)]


def postprocess_decoded(decoded_sentence, input_sentence, attention_scores):
    if attention_scores is None:
        result = decoded_sentence.split()
    else:
        source_sentence_tokenized = src_tokenizer(input_sentence)
        max_input_sentence_length = len(source_sentence_tokenized)
        max_decode_length = attention_scores.size(0)
        decoded_sentence_tokens = decoded_sentence.split()
        result = []
        for tgt_id, tgt_token in enumerate(decoded_sentence_tokens):
            if tgt_token == cfg.unk_token and tgt_id < max_decode_length:
                input_sentence_id = int(attention_scores[tgt_id].item())
                if input_sentence_id < max_input_sentence_length:
                    lex = source_sentence_tokenized[input_sentence_id]
                    # TODO try lexical translation first
                    result.append(lex)
                    continue
            result.append(tgt_token)
    return tgt_detokenizer(result)


def evaluate(data_iter: data.BucketIterator, dp: DataProvider, model: nn.Module, src_file: str, gold_tgt_file: str, src_sgm: str, gold_tgt_sgm: str,
             eph: str, save_decoded_sentences: bool = False, output_dir: str = '../.output', nuance: str = '0000000000'):
    if not save_decoded_sentences:
        print("Evaluation ....")
    model.eval()
    result_file = None
    cp_name = cfg.checkpoint_name
    if cp_name.endswith(".pt"):
        cp_name = cp_name[:-3]
    cp_name += "_" + nuance
    if save_decoded_sentences:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(os.path.join(output_dir, cp_name)):
            os.mkdir(os.path.join(output_dir, cp_name))
        output_path = os.path.join(output_dir, cp_name, data_iter.dataset.name)
        print("Storing decoding results for {} in {}".format(data_iter.dataset.name, os.path.abspath(output_path)))
        result_file = open(output_path, "w", encoding='utf-8')

    def _get_next_line(file_1_iter, file_2_iter):
        """
        torchtext.data.BucketIterator ignores the parallel lines one side of which is an empty sequenece.
        We need this function to keep our original files in par with torchtext approach.
        """
        for l1, l2 in zip(file_1_iter, file_2_iter):
            l1 = l1.strip()
            l2 = l2.strip()
            if not len(l1) or not len(l2):
                if save_decoded_sentences:
                    result_file.write("\n")
                continue
            yield l1, l2

    src_originals = iter(open(src_file, "r"))
    tgt_originals = iter(open(gold_tgt_file, "r"))
    originals = _get_next_line(src_originals, tgt_originals)
    random_sample_created = False
    with torch.no_grad():
        lall_valid = 0.0
        lcount_valid = 0.0
        all_bleu_score = 0.0
        sent_count = 0.0
        for valid_instance in data_iter:
            pred, max_attention_idcs, lss, _, n_tokens = model(valid_instance.src, valid_instance.trg, test_mode=True, **valid_instance.data_args)
            lall_valid += lss.item()
            lcount_valid += n_tokens
            for d_id, (decoded, model_expected) in enumerate(zip(
                    convert_target_batch_back(pred, dp.TGT), convert_target_batch_back(valid_instance.trg[0], dp.TGT))):
                source_sentence, reference_sentence = next(originals)
                if bool(cfg.lowercase_data):
                    source_sentence = source_sentence.lower()
                    reference_sentence = reference_sentence.lower()
                decoded = postprocess_decoded(decoded, source_sentence, max_attention_idcs.select(1, d_id)
                                              if max_attention_idcs is not None else None)
                all_bleu_score += sacrebleu.corpus_bleu([decoded], [[reference_sentence]]).score
                if save_decoded_sentences:
                    result_file.write(decoded+"\n")
                sent_count += 1.0
                if not save_decoded_sentences and not random_sample_created and random.random() < 0.01:
                    random_sample_created = True
                    try:
                        print("Sample Inp't: {}\nSample Pred : {}\nModel Expc'd: {}\nSample Act'l: {}".format(
                            source_sentence, decoded, model_expected, reference_sentence))
                    except UnicodeEncodeError:  # some sentences in the raw file might not be nicely formatted!
                        random_sample_created = False
        average_loss = lall_valid/max(lcount_valid, 1)
        average_bleu = all_bleu_score / max(sent_count, 1)
        if average_loss > 0.0:
            print("E {} ::: Average Loss {:.2f} ::: Average BleuP1 [C04-1072] {:.2f}".format(eph, average_loss, average_bleu))
        else:
            print("E {} ::: Average BleuP1 [C04-1072] {:.2f}".format(eph, average_bleu))
    model.train()
    if save_decoded_sentences:
        result_file.close()
        # TODO perform recasing here if lowercased!
        recased_path = output_path
        p = subprocess.Popen(["perl", "./scripts/wrap-xml.perl", dp.processed_data.target_language, src_sgm, cp_name[:-3]],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        with open(recased_path, "r", encoding='utf-8') as reacsed:
            p.stdin.write(reacsed.read().encode('utf-8'))
            p.stdin.close()
            with open(recased_path+".sgm", "w", encoding='utf-8') as reacsed_sgm:
                the_output = p.stdout.read()
                reacsed_sgm.write(the_output.decode('utf-8'))
        if bool(cfg.lowercase_data):
            mteval = subprocess.check_output(
                ["perl", "scripts/mteval-v14.pl", "-r", gold_tgt_sgm, "-s", src_sgm, "-t", recased_path+".sgm"]).decode('utf-8')
        else:
            mteval = subprocess.check_output(
                ["perl", "scripts/mteval-v14.pl", "-r", gold_tgt_sgm, "-s", src_sgm, "-t", recased_path+".sgm", "-c"]).decode('utf-8')
        score_line = mteval.split("\n")[9].split(" ")
        nist_score = float(score_line[3])
        bleu_score = float(score_line[8]) * 100
        print("E {} ::: BLEU Score [mteval] {:.2f} ::: NIST Score [mteval] {:.2f}".format(eph, bleu_score, nist_score))
    return average_loss, average_bleu
