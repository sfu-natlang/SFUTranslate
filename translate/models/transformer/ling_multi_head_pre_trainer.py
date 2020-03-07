"""
The test script for training an intermediary sub-sectioned layer which does contain the same exact information as bert

"""
import sys
import spacy
from spacy.tokenizer import Tokenizer
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as init
import torch
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from textblob import TextBlob
from nltk.wsd import lesk
import unidecode
from random import random

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from transformers import BertTokenizer, BertForMaskedLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ###############################################CONFIGURATIONS########################################################
model_name = 'bert-base-uncased'
number_of_bert_layers = 13
D_in, H, D_out = 768, 1024, 768
epochs = 10
lr = 0.05
batch_size = 32
max_norm = 5
scheduler_patience_steps = 60
scheduler_min_lr = 0.001
scheduler_decay_factor = 0.9

# ###############################################REQUIRED FUNCTIONS####################################################


def weight_init(m):
    """
    The neural network initializer function
    :param m: nn parameter set
    """
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)


def get_next_batch(file_adr, b_size):
    """
    :param file_adr: the address of the train data (e.g. "./.data/multi30k/train.en")
    :param b_size: teh batch size of the returning sentence batches
    """
    res = []
    for l in Path(file_adr).open("r"):
        res.append(l)
        if len(res) == b_size:
            yield res
            del res[:]
    yield res


def find_token_index_in_list(spacy_token, tokens_doc, check_lowercased_doc_tokens=False):
    if spacy_token is None or tokens_doc is None or not len(tokens_doc):
        return []
    if check_lowercased_doc_tokens:
        inds = [i for i, val in enumerate(tokens_doc) if check_tokens_equal(
            spacy_token, val) or check_tokens_equal(spacy_token, val.lower())]
    else:
        inds = [i for i, val in enumerate(tokens_doc) if check_tokens_equal(spacy_token, val)]
    # assert len(inds) == 1
    return inds


def check_tokens_equal(spacy_token, bert_token):
    if spacy_token is None:
        return bert_token is None
    if bert_token is None:
        return spacy_token is None
    if bert_token == spacy_token.lower() or bert_token == spacy_token:
        return True
    spacy_token = unidecode.unidecode(spacy_token)  # remove accents and unicode emoticons
    bert_token = unidecode.unidecode(bert_token)  # remove accents and unicode emoticons
    if bert_token == spacy_token.lower() or bert_token == spacy_token:
        return True
    return False


def check_for_inital_subword_sequence(spacy_doc, bert_doc):
    seg_s_i = 0
    spacy_token = spacy_doc[seg_s_i]
    bert_f_pointer = 0
    bert_token = bert_doc[bert_f_pointer]
    while not check_tokens_equal(spacy_token, bert_token):
        bert_f_pointer += 1
        if bert_f_pointer < len(bert_doc):
            tmp = bert_doc[bert_f_pointer]
        else:
            bert_f_pointer = 0
            seg_s_i = -1
            break
        bert_token += tmp[2:] if tmp.startswith("##") else tmp
    return seg_s_i, bert_f_pointer


def spacy_to_bert_aligner(spacy_doc, bert_doc, print_alignments=False, level=0):
    """
    This function receives two lists extracted from spacy, and bert tokenizers on the same sentence,
    and returns the alignment fertilities from spacy to bert.
    the output will have a length equal to the size of spacy_doc each index of which indicates the number
    of times the spacy element characteristics must be copied to equal the length of the bert tokenized list.
    This algorithm enforces the alignments in a strictly left-to-right order.
    """
    previous_spacy_token = None
    sp_len = len(spacy_doc)
    bt_len = len(bert_doc)
    if not sp_len:
        return []
    elif not bt_len:
        return [0] * len(spacy_doc)
    elif sp_len == 1:  # one to one and one to many
        return [bt_len]
    elif bt_len == 1:  # many to one case
        r = [0] * sp_len
        r[0] = 1
        return r
    # many to many case is being handled in here:
    seg_s_i = -1
    seg_bert_f_pointer = -1
    best_right_candidate = None
    best_left_candidate = None
    for s_i in range(sp_len):
        spacy_token = spacy_doc[s_i]
        next_spacy_token = spacy_doc[s_i + 1] if s_i < len(spacy_doc) - 1 else None
        prev_eq = None
        current_eq = None
        next_eq = None
        previous_bert_token = None
        exact_expected_location_range_list = find_token_index_in_list(spacy_token, bert_doc)
        if not len(exact_expected_location_range_list):
            exact_expected_location_range = -1
        elif len(exact_expected_location_range_list) == 1:
            exact_expected_location_range = exact_expected_location_range_list[0]
        else:  # multiple options to choose from
            selection_index_list = find_token_index_in_list(spacy_token, spacy_doc, check_lowercased_doc_tokens=True)
            # In cases like [hadn 't and had n't] or wrong [UNK] merges:
            #       len(exact_expected_location_range_list) < len(selection_index_list)
            # In cases like punctuations which will get separated in bert tokenizer and don't in spacy or subword breaks
            #       len(exact_expected_location_range_list) > len(selection_index_list)
            selection_index = selection_index_list.index(s_i)
            if selection_index < len(exact_expected_location_range_list):
                # TODO account for distortion (if some other option has less distortion take it)
                exact_expected_location_range = exact_expected_location_range_list[selection_index]
            else:
                raise ValueError("selection_index is greater than the available list")
        end_of_expected_location_range = exact_expected_location_range+1 if exact_expected_location_range > -1 else s_i+len(spacy_token)+2
        start_of_expected_location_range = exact_expected_location_range - 1 if exact_expected_location_range > -1 else s_i-1
        for bert_f_pointer in range(
                max(start_of_expected_location_range, 0), min(len(bert_doc), end_of_expected_location_range)):
            bert_token = bert_doc[bert_f_pointer]
            next_bert_token = bert_doc[bert_f_pointer + 1] if bert_f_pointer < len(bert_doc) - 1 else None
            prev_eq = check_tokens_equal(previous_spacy_token, previous_bert_token)
            current_eq = check_tokens_equal(spacy_token, bert_token)
            next_eq = check_tokens_equal(next_spacy_token, next_bert_token)
            if prev_eq and current_eq and next_eq:
                seg_bert_f_pointer = bert_f_pointer
                break
            elif prev_eq and current_eq and best_left_candidate is None:
                best_left_candidate = (s_i, bert_f_pointer)
            elif current_eq and next_eq and best_right_candidate is None:
                best_right_candidate = (s_i, bert_f_pointer)
            previous_bert_token = bert_token
        if prev_eq and current_eq and next_eq:
            seg_s_i = s_i
            break
        previous_spacy_token = spacy_token

    curr_fertilities = [1]
    if seg_s_i == -1 or seg_bert_f_pointer == -1:
        if best_left_candidate is not None and best_right_candidate is not None:  # accounting for min distortion
            seg_s_i_l, seg_bert_f_pointer_l = best_left_candidate
            seg_s_i_r, seg_bert_f_pointer_r = best_right_candidate
            if seg_bert_f_pointer_r - seg_s_i_r < seg_bert_f_pointer_l - seg_s_i_l:
                seg_s_i, seg_bert_f_pointer = best_right_candidate
            else:
                seg_s_i, seg_bert_f_pointer = best_left_candidate
        elif best_left_candidate is not None:
            seg_s_i, seg_bert_f_pointer = best_left_candidate
        elif best_right_candidate is not None:
            seg_s_i, seg_bert_f_pointer = best_right_candidate
        else:  # multiple subworded tokens stuck together
            seg_s_i, seg_bert_f_pointer = check_for_inital_subword_sequence(spacy_doc, bert_doc)
            curr_fertilities = [seg_bert_f_pointer + 1]
    if seg_s_i == -1 or seg_bert_f_pointer == -1 and len(spacy_doc[0]) < len(bert_doc[0]): # none identical tokenization
        seg_s_i = 0
        seg_bert_f_pointer = 0
    if seg_s_i == -1 or seg_bert_f_pointer == -1:
        print(spacy_doc)
        print(bert_doc)
        raise ValueError()
    if seg_s_i > 0:  # seg_bert_f_pointer  is always in the correct range
        left = spacy_to_bert_aligner(spacy_doc[:seg_s_i], bert_doc[:seg_bert_f_pointer], False, level+1)
    else:
        left = []
    if seg_s_i < sp_len:  # seg_bert_f_pointer  is always in the correct range
        right = spacy_to_bert_aligner(spacy_doc[seg_s_i+1:], bert_doc[seg_bert_f_pointer+1:], False, level+1)
    else:
        right = []
    fertilities = left + curr_fertilities + right
    if print_alignments and not level:
        bert_ind = 0
        for src_token, fertility in zip(spacy_doc, fertilities):
            for b_f in range(fertility):
                print("{} --> {}".format(src_token, bert_doc[bert_ind+b_f]))
            bert_ind += fertility
    if not level and sum(fertilities) != len(bert_doc):
        print("Warning one sentence is not aligned properly:\n{}\n{}\n{}\n{}".format(
            spacy_doc, bert_doc, sum(fertilities), len(bert_doc)))
    return fertilities


def extract_linguistic_features(line, bert_tokenizer, extract_sentiment=False):
    result = []
    lesk_queries = {"NOUN": 'n', "VERB": 'v', "ADJ": 'a', "ADV": 'r'}
    doc = nlp(line)
    spacy_doc = [token.text for token in doc]
    bert_doc = [token for token in bert_tokenizer.tokenize(line)]
    sp_bert_doc = sp_bert(" ".join(bert_doc))
    spacy_labeled_bert_tokens = [token.text for token in sp_bert_doc]
    assert len(spacy_labeled_bert_tokens) == len(bert_doc), "{}\n{}".format(spacy_labeled_bert_tokens, bert_doc)
    fertilities = spacy_to_bert_aligner(spacy_doc, bert_doc)
    bert_doc_pointer = 0
    for token, fertility in zip(doc, fertilities):
        pos = token.pos_
        tag = token.tag_ if len(token.tag_) else "NONE"
        shape = token.shape_
        ent_type = token.ent_type_ if len(token.ent_type_) else "NONE"
        ent_iob = token.ent_iob_ if len(token.ent_type_) else "O"
        sense_data = lesk(spacy_doc, token.text, lesk_queries[pos] if pos in lesk_queries else "")
        sense = sense_data.name().split(".")[-1] if sense_data is not None else "none"
        if extract_sentiment:
            sentiment = 'positive' if TextBlob(token.text).sentiment.polarity > 0.05 else 'negative' \
                if TextBlob(token.text).sentiment.polarity < -0.05 else 'none'
        else:
            sentiment = "none"
        linguistic_features = {"pos": pos, "tag": tag, "shape": unidecode.unidecode(shape), "ent_type": ent_type,
                               "ent_iob": ent_iob, "sense": sense, "sentiment": sentiment}
        for f_index in range(fertility):
            if fertility == 1:
                bis = "s_"  # short for "single"
            elif f_index == 1:
                bis = "b_"  # short for "begin"
            else:
                bis = "i_"  # short for "inside"
            lfc = linguistic_features.copy()
            lfc["token"] = bert_doc[bert_doc_pointer]
            lfc["shape"] = unidecode.unidecode(sp_bert_doc[bert_doc_pointer].shape_)
            lfc["pos"] = bis + lfc["pos"]
            lfc["tag"] = bis + lfc["tag"]
            result.append(lfc)
            bert_doc_pointer += 1
        del linguistic_features
    return result


def projection_trainer(file_adr, bert_tokenizer):
    bert_lm = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    model = torch.nn.Sequential(nn.Linear(D_in, H), nn.Linear(H, D_out)).to(device)

    model.apply(weight_init)
    bert_weights_for_average_pooling = nn.Parameter(torch.zeros(number_of_bert_layers), requires_grad=True)
    softmax = nn.Softmax(dim=-1)
    opt = optim.SGD(model.parameters(), lr=float(lr))
    loss_fn = torch.nn.MSELoss(reduction='sum')
    for t in range(epochs):
        all_loss = 0.0
        all_tokens_count = 0.0
        itr = tqdm(get_next_batch(file_adr, batch_size))
        for input_sentences in itr:
            sequences = [torch.tensor(bert_tokenizer.encode(input_sentence, add_special_tokens=True), device=device)
                         for input_sentence in input_sentences]
            input_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            outputs = bert_lm(input_ids, masked_lm_labels=input_ids)[2]  # (batch_size * [input_length + 2] * 768)
            all_layers_embedded = torch.cat([o.detach().unsqueeze(0) for o in outputs], dim=0)
            embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0), softmax(bert_weights_for_average_pooling))
            for s in range(embedded.size(1)):
                x = embedded.select(1, s)
                pred = model(x)
                loss = loss_fn(pred, x)
                model.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                opt.step()
                all_loss += loss.item()
                all_tokens_count += x.size(0)
                itr.set_description("Epoch: {}, Average Loss: {:.2f}".format(t, all_loss / all_tokens_count))


def create_empty_linguistic_vocab():
    return {"pos": {}, "tag": {}, "shape": {}, "ent_type": {}, "ent_iob": {}, "sense": {}, "sentiment": {}}


def extract_linguistic_vocabs(file_adr, bert_tokenizer):
    vocabs = create_empty_linguistic_vocab()
    vocab_cnts = {"pos": Counter(), "tag": Counter(), "shape": Counter(), "ent_type": Counter(), "ent_iob": Counter(), "sense": Counter(), "sentiment": Counter()}
    vocab_totals = {"pos": 0., "tag": 0., "shape": 0., "ent_type": 0., "ent_iob": 0., "sense": 0., "sentiment": 0.}
    itr = tqdm(get_next_batch(file_adr, batch_size))
    for input_sentences in itr:
        for sent in input_sentences:
            res = extract_linguistic_features(sent, bert_tokenizer)
            for res_item in res:
                for key in res_item:
                    value = res_item[key]
                    if key in vocabs and value not in vocabs[key]:
                        vocabs[key][value] = len(vocabs[key])
                    if key in vocab_cnts:
                        vocab_cnts[key][value] += 1.
                    if key in vocab_totals:
                        vocab_totals[key] += 1.
    vs = dict([(key, dict([(key2, (vocabs[key][key2], vocab_cnts[key][key2]/vocab_totals[key]))
                           for key2 in vocab_cnts[key]])) for key in vocab_cnts])
    return vs


class SubLayerED(torch.nn.Module):
    def __init__(self, D_in, Hs, D_out, feature_sizes, padding_index=0):
        super(SubLayerED, self).__init__()
        self.equal_length_Hs = (sum([Hs[0] == h for h in Hs]) == len(Hs))
        self.consider_adversarial_loss = False
        self.encoders = nn.ModuleList([nn.Linear(D_in, h) for h in Hs])
        self.decoder = nn.Linear(sum(Hs), D_out)
        self.feature_classifiers = nn.ModuleList([nn.Linear(h, o) for h, o in zip(Hs[:-1], feature_sizes)])

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.class_loss_fn = nn.CrossEntropyLoss(ignore_index=padding_index, reduction='none')
        self.pair_distance = nn.PairwiseDistance(p=2)
        self.discriminator = nn.Linear(D_out, 1)

        self.bert_weights_for_average_pooling = nn.Parameter(torch.zeros(number_of_bert_layers), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        self.verbose_debug_percentage = 0.0001

    def verbose_results(self, ling_classes, features, feature_weights):
        if random() < self.verbose_debug_percentage:  # debug
            print("\n")
            for ind, lc in enumerate(ling_classes):
                class_type = features_list[ind]  # TODO remove this line it is not safe
                true_labels = features[ind]
                predicted_labels = lc.argmax(dim=-1)
                for e in range(true_labels.size(0)):
                    tl = true_labels[e].item()
                    if tl > 0:
                        true_label = reverse_linguistic_vocab[class_type][tl - 1]
                    else:
                        continue
                    pl = predicted_labels[e].item()
                    if pl > 0:
                        pred_label = reverse_linguistic_vocab[class_type][pl - 1]
                    else:
                        pred_label = '[PAD]'
                    if true_label != pred_label:
                        print("Mistaken prediction of {}.{} instead of {}.{} (w: {:.3f})".format(
                            class_type, pred_label, class_type, true_label, feature_weights[ind][e].item()))
                    # else:
                    #    print("Correctly predicted: {}.{} (w: {:.3f})".format(
                    #        class_type, true_label, feature_weights[ind][e].item()))

    def forward(self, x, features, feature_weights):
        encoded = [self.encoders[i](x) for i in range(len(self.encoders))]
        y_pred = self.decoder(torch.cat(encoded, 1))
        ling_classes = [self.feature_classifiers[i](encoded[i]) for i in range(len(self.encoders)-1)]
        loss = self.loss_fn(y_pred, x)
        # feature_pred_correct = [(lc.argmax(dim=-1) == features[ind]) for ind, lc in enumerate(ling_classes)]
        self.verbose_results(ling_classes, features, feature_weights)
        feature_pred_correct = []
        for ind, lc in enumerate(ling_classes):
            wrong_or_pad = lc.argmax(dim=-1) == features[ind]
            res = []
            for e in range(wrong_or_pad.size(0)):
                if features[ind][e] == 0:
                    continue  # Leave pad indices out of accuracy calculation
                else:
                    res.append(wrong_or_pad[e])
            if len(res):
                feature_pred_correct.append(torch.cat([r.unsqueeze(0) for r in res]))
            else:
                feature_pred_correct.append(torch.empty(0))
        for ind, lc in enumerate(ling_classes):
            mask = (feature_weights[ind] != 0.).float()
            c_loss = self.class_loss_fn(lc, features[ind])
            w = feature_weights[ind]
            final_c_loss = (c_loss * mask) / (w+1e-32)
            loss += final_c_loss.sum()
        if self.equal_length_Hs:
            # for the case of equal lengths this part makes sure that the vectors are different from each other
            lss = 0
            cnt = 0.0
            for i in range(len(self.encoders)-1):  # see questions in LingEmb document for this part!
                for j in range(i+1, len(self.encoders)-1):
                    l = self.pair_distance(encoded[i], encoded[j])
                    cnt += l.size(0)  # l is a one-dimensional tensor of size batch_size
                    lss += l.sum()
            if cnt > 0.0:
                loss += lss / cnt
        if self.consider_adversarial_loss:
            # Adversarial loss
            fake_pred = self.discriminator(torch.empty(y_pred.size()).normal_(mean=0, std=1.0).to(device))
            true_pred = self.discriminator(y_pred)
            l_true = self.loss_fn(true_pred, torch.ones(true_pred.size(), device=device))
            l_fake = self.loss_fn(fake_pred, torch.zeros(fake_pred.size(), device=device))
            loss += l_true / true_pred.view(-1).size(0) + l_fake / fake_pred.view(-1).size(0)
        return y_pred, loss, feature_pred_correct


def map_sentences_to_vocab_ids(input_sentences, required_features_list, linguistic_vocab, bert_tokenizer):
    padding_value = 0
    extracted_features = [[] for _ in required_features_list]
    extracted_feature_weights = [[] for _ in required_features_list]
    for sent in input_sentences:
        sent_extracted_features = [[padding_value] for _ in required_features_list]
        sent_extracted_feature_weights = [[0.] for _ in required_features_list]
        res = extract_linguistic_features(sent, bert_tokenizer)
        for token_feature in res:
            for ind, elem in enumerate(required_features_list):
                assert elem in token_feature, "feature {} is required to be extracted!"
                feature = token_feature[elem]
                feature_id = linguistic_vocab[elem][feature][0] + 1  # the first index is <PAD>
                feature_weight = linguistic_vocab[elem][feature][1]
                sent_extracted_features[ind].append(feature_id)
                sent_extracted_feature_weights[ind].append(feature_weight)
        for ind in range(len(required_features_list)):
            sent_extracted_features[ind].append(padding_value)
            sent_extracted_feature_weights[ind].append(0.)  # make sure you don't sum over this one
            extracted_features[ind].append(torch.tensor(sent_extracted_features[ind], device=device).long())
            extracted_feature_weights[ind].append(torch.tensor(sent_extracted_feature_weights[ind], device=device).float())

    return [torch.nn.utils.rnn.pad_sequence(extracted_features[ind], batch_first=True, padding_value=padding_value)
            for ind in range(len(required_features_list))], \
           [torch.nn.utils.rnn.pad_sequence(extracted_feature_weights[ind], batch_first=True, padding_value=0.)
            for ind in range(len(required_features_list))]


def project_sub_layers_trainer(file_adr, bert_tokenizer, linguistic_vocab, required_features_list,
                               save_model_name="project_sublayers.pt", relative_sizing=False):
    """
    Implementation of the sub-layer model trainer which pre-trains the transformer heads using the BERT vectors.
    """
    assert len(required_features_list) > 0, "You have to select some features"
    assert linguistic_vocab is not None and len(linguistic_vocab) > 0

    Hs = []
    for rfl in required_features_list:
        if rfl in linguistic_vocab:
            if relative_sizing:
                print("This might not be supported in the multi-head implementation")
                Hs.append(len(linguistic_vocab[rfl]))
            else:
                # TODO consider hierarchical encoding of features here
                Hs.append(1.0)
    Hs.append(max(Hs))
    weight_ratio = int(float(H)/sum(Hs))
    assert weight_ratio > 1
    Hs = [int(weight_ratio * ind) for ind in Hs]
    Hs[-1] += max(0, (H - sum(Hs)))

    bert_lm = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    model = SubLayerED(D_in, Hs, D_out, [len(linguistic_vocab[f]) + 1 for f in required_features_list]).to(device)
    model.apply(weight_init)
    opt = optim.SGD(model.parameters(), lr=float(lr), momentum=0.9)
    scheduler = ReduceLROnPlateau(opt, mode='min', patience=scheduler_patience_steps, factor=scheduler_decay_factor,
                                  threshold=0.001, verbose=True, min_lr=scheduler_min_lr)
    for t in range(epochs):
        all_loss = 0.0
        all_tokens_count = 0.0
        itr = tqdm(get_next_batch(file_adr, batch_size))
        feature_pred_corrects = [0 for _ in range(len(required_features_list))]
        feature_pred_correct_all = 0.0
        for input_sentences in itr:
            sequences = [torch.tensor(bert_tokenizer.encode(input_sentence, add_special_tokens=True), device=device)
                         for input_sentence in input_sentences]
            features, feature_weights = map_sentences_to_vocab_ids(
                input_sentences, required_features_list, linguistic_vocab, bert_tokenizer)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=bert_tokenizer.pad_token_id)
            outputs = bert_lm(input_ids, masked_lm_labels=input_ids)[2]  # (batch_size * [input_length + 2] * 768)
            all_layers_embedded = torch.cat([o.detach().unsqueeze(0) for o in outputs], dim=0)
            embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0),
                                    model.softmax(model.bert_weights_for_average_pooling))
            for s in range(1, embedded.size(1)-1):
                x = embedded.select(1, s)
                features_selected = []
                feature_weights_selected = []
                permitted_to_continue = True
                for f, fw in zip(features, feature_weights):
                    if s < f.size(1):
                        features_selected.append(f.select(1, s))
                        feature_weights_selected.append(fw.select(1, s))
                    else:
                        permitted_to_continue = False
                if not permitted_to_continue:
                    continue
                _, loss, feature_pred_correct = model(x, features_selected, feature_weights_selected)
                for ind, score in enumerate(feature_pred_correct):
                    feature_pred_corrects[ind] += score.sum().item()
                feature_pred_correct_all += feature_pred_correct[0].size(0)
                model.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                opt.step()
                all_loss += loss.item()
                all_tokens_count += x.size(0)
                classification_report = ["{}:{:.2f}%".format(feat.upper(), float(feature_pred_corrects[ind] * 100)
                                         / feature_pred_correct_all) for ind, feat in enumerate(required_features_list)]
                itr.set_description("Epoch: {}, Average Loss: {:.2f}, [{}]".format(t, all_loss / all_tokens_count,
                                                                                   "; ".join(classification_report)))
            scheduler.step(all_loss / all_tokens_count)
        for ind, feat in enumerate(required_features_list):
            print("{} classification precision: {:.2f}%".format(
                feat.upper(), float(feature_pred_corrects[ind] * 100) / feature_pred_correct_all))
        torch.save({'model': model}, save_model_name)


def project_sub_layers_tester(input_sentences, bert_tokenizer, linguistic_vocab, required_features_list,
                              load_model_name="project_sublayers.pt"):
    bert_lm = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    saved_obj = torch.load(load_model_name, map_location=lambda storage, loc: storage)
    model = saved_obj['model'].to(device)
    feature_pred_corrects = [0 for _ in range(len(required_features_list))]
    feature_pred_correct_all = 0.0
    with torch.no_grad():
        all_loss = 0.0
        all_tokens_count = 0.0
        sequences = [torch.tensor(bert_tokenizer.encode(input_sentence), device=device)
                     for input_sentence in input_sentences]
        features, feature_weights = map_sentences_to_vocab_ids(
            input_sentences, required_features_list, linguistic_vocab, bert_tokenizer)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=bert_tokenizer.pad_token_id)
        outputs = bert_lm(input_ids, masked_lm_labels=input_ids)[2]  # (batch_size * [input_length + 2] * 768)
        all_layers_embedded = torch.cat([o.detach().unsqueeze(0) for o in outputs], dim=0)
        embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0),
                                model.softmax(model.bert_weights_for_average_pooling))
        for s in range(1, embedded.size(1)-1):
            x = embedded.select(1, s)
            features_selected = [f.select(1, s) for f in features]
            feature_weights_selected = [fw.select(1, s) for fw in feature_weights]
            _, loss, feature_pred_correct = model(x, features_selected, feature_weights_selected)
            for ind, score in enumerate(feature_pred_correct):
                feature_pred_corrects[ind] += score.sum().item()
            feature_pred_correct_all += feature_pred_correct[0].size(0)
            all_loss += loss.item()
            all_tokens_count += x.size(0)
    for ind, feat in enumerate(required_features_list):
        print("{} classification precision: {:.2f}%".format(
            feat.upper(), float(feature_pred_corrects[ind] * 100) / feature_pred_correct_all))
    print(("Average Test Loss: {:.2f}".format(all_loss / all_tokens_count)))


if __name__ == '__main__':
    nlp = spacy.load("en")
    sp_bert = spacy.load("en")
    sp_bert.tokenizer = Tokenizer(sp_bert.vocab)
    bert_tknizer = BertTokenizer.from_pretrained(model_name)
    running_mode = int(sys.argv[1])
    if running_mode == 0:
        projection_trainer(sys.argv[2], bert_tknizer)
    elif running_mode == 1:
        batch_size = 256
        print(extract_linguistic_vocabs(sys.argv[2], bert_tknizer))
    elif running_mode == 2:
        multi30k_linguistic_vocab = {'pos': {'s_NUM': (0, 0.017723759603866726), 's_ADJ': (1, 0.07492351240205392), 's_PUNCT': (2, 0.08467679380266073), 's_NOUN': (3, 0.2685979944099837), 's_AUX': (4, 0.03135292127944741), 's_ADV': (5, 0.010511163501618764), 's_SCONJ': (6, 0.009516761147606441), 's_ADP': (7, 0.1319121772325096), 's_VERB': (8, 0.09762691346156292), 's_DET': (9, 0.17667808576275237), 'i_NOUN': (10, 0.01972782368305269), 'b_NOUN': (11, 0.014857541054066494), 'i_ADJ': (12, 0.002642414439434284), 'b_ADJ': (13, 0.0019913479365515348), 's_PART': (14, 0.003985239101629955), 's_CCONJ': (15, 0.019348882632546714), 'i_VERB': (16, 0.00645980045828978), 'b_VERB': (17, 0.005185642966320025), 's_PRON': (18, 0.007416054384398819), 's_PROPN': (19, 0.007169361217290902), 'i_PROPN': (20, 0.002642414439434284), 'b_PROPN': (21, 0.0018006057970351042), 'i_PART': (22, 0.001030007553388725), 'b_PART': (23, 0.001030007553388725), 'i_AUX': (24, 0.00013733434045182997), 'b_AUX': (25, 0.00013733434045182997), 'i_ADV': (26, 0.00036368167934466085), 'b_ADV': (27, 0.0002797551379574314), 's_INTJ': (28, 3.814842790328611e-05), 'i_DET': (29, 2.5432285268857402e-06), 'b_DET': (30, 2.5432285268857402e-06), 's_X': (31, 3.3061970849514626e-05), 'i_NUM': (32, 5.340779906460055e-05), 'b_NUM': (33, 3.0518742322628884e-05), 's_SYM': (34, 2.7975513795743143e-05), 'i_PRON': (35, 1.0172914107542961e-05), 'b_PRON': (36, 7.629685580657221e-06), 'i_ADP': (37, 1.7802599688200184e-05), 'b_ADP': (38, 1.5259371161314442e-05), 'i_PUNCT': (39, 1.2716142634428702e-05), 'b_PUNCT': (40, 5.0864570537714805e-06), 'i_X': (41, 5.0864570537714805e-06), 'b_X': (42, 5.0864570537714805e-06), 'i_SPACE': (43, 5.0864570537714805e-06), 'b_SPACE': (44, 2.5432285268857402e-06)}, 'tag': {'s_CD': (0, 0.017723759603866726), 's_JJ': (1, 0.07346624245614838), 's_,': (2, 0.01007881465204819), 's_NNS': (3, 0.05108583141955387), 's_VBP': (4, 0.016785308277445888), 's_RB': (5, 0.010249210963349534), 's_IN': (6, 0.1355286481977411), 's_.': (7, 0.07032789845397137), 's_VBG': (8, 0.061368104353752916), 's_DT': (9, 0.16525644644850854), 'i_NN': (10, 0.01205490321743841), 'b_NN': (11, 0.009203944038799495), 's_NN': (12, 0.2188727902523137), 's_VBZ': (13, 0.03550092700679805), 's_PRP$': (14, 0.01021614899250002), 's_VBN': (15, 0.00816122034277634), 'i_JJ': (16, 0.002604266011530998), 'b_JJ': (17, 0.0019659156512826775), 's_CC': (18, 0.019348882632546714), 'i_VBG': (19, 0.0033265429131665485), 'b_VBG': (20, 0.002566117583627712), 's_TO': (21, 0.0038758802749738683), 's_WDT': (22, 0.0011597122082598976), 'i_NNS': (23, 0.007672920465614279), 'b_NNS': (24, 0.005653597015267001), 's_PRP': (25, 0.004600700405136304), 's_RP': (26, 0.005900290182374918), 's_NNP': (27, 0.006912495136075442), 'i_NNP': (28, 0.0025610311265739406), 'b_NNP': (29, 0.0017268521697554177), 'i_VBN': (30, 0.001075785666872668), 'b_VBN': (31, 0.0008570680135604945), 's_VB': (32, 0.005694288671697172), 'i_POS': (33, 0.0009918591254854388), 'b_POS': (34, 0.0009918591254854388), 'i_VBZ': (35, 0.0016912469703790174), 'b_VBZ': (36, 0.0015132209734970154), 's_WP': (37, 0.0008240060427109799), 's_HYPH': (38, 0.003301110627897691), 's_JJR': (39, 0.0014165782894753575), 's_VBD': (40, 0.001284330406077299), 's_EX': (41, 0.0006307206746676636), 'i_RB': (42, 0.0003942004216672898), 'b_RB': (43, 0.00031281710880694605), 'i_JJR': (44, 3.560519937640037e-05), 'b_JJR': (45, 2.2889056741971663e-05), 's_``': (46, 0.0003942004216672898), 'i_NNPS': (47, 8.138331286034369e-05), 'b_NNPS': (48, 7.375362727968647e-05), "s_\''": (49, 0.0003687681363984324), 's_WRB': (50, 0.00030010096617251735), 's_MD': (51, 0.00018565568246265904), 'i_VBP': (52, 0.0001703963113013446), 'b_VBP': (53, 0.00014242079750560147), 's_-LRB-': (54, 5.340779906460055e-05), 's_-RRB-': (55, 5.086457053771481e-05), 's_:': (56, 0.00010172914107542962), 'i_VB': (57, 0.00019837182509708774), 'b_VB': (58, 0.00014750725455937294), 's_PDT': (59, 3.814842790328611e-05), 'i_VBD': (60, 0.0001322478833980585), 'b_VBD': (61, 9.409945549477239e-05), 's_NNPS': (62, 0.0002568660812154598), 's_UH': (63, 3.814842790328611e-05), 's_POS': (64, 3.560519937640037e-05), 'i_DT': (65, 2.5432285268857402e-06), 'b_DT': (66, 2.5432285268857402e-06), 's_RBS': (67, 2.7975513795743143e-05), 's_WP$': (68, 7.629685580657221e-06), 's_FW': (69, 1.0172914107542961e-05), 'i_CD': (70, 5.340779906460055e-05), 'b_CD': (71, 3.0518742322628884e-05), 's_$': (72, 2.7975513795743143e-05), 'i_PRP': (73, 1.0172914107542961e-05), 'b_PRP': (74, 7.629685580657221e-06), 's_JJS': (75, 4.0691656430171844e-05), 's_RBR': (76, 7.629685580657221e-06), 's_LS': (77, 2.0345828215085922e-05), 'i_IN': (78, 1.7802599688200184e-05), 'b_IN': (79, 1.5259371161314442e-05), 'i_.': (80, 2.5432285268857402e-06), 'b_.': (81, 2.5432285268857402e-06), 's_XX': (82, 2.5432285268857402e-06), 'i_XX': (83, 2.5432285268857402e-06), 'b_XX': (84, 2.5432285268857402e-06), 'i_FW': (85, 2.5432285268857402e-06), 'b_FW': (86, 2.5432285268857402e-06), 'i_JJS': (87, 2.5432285268857402e-06), 'b_JJS': (88, 2.5432285268857402e-06), 'i_:': (89, 1.0172914107542961e-05), 'b_:': (90, 2.5432285268857402e-06), 'i__SP': (91, 5.0864570537714805e-06), 'b__SP': (92, 2.5432285268857402e-06), 'i_RBS': (93, 5.0864570537714805e-06), 'b_RBS': (94, 2.5432285268857402e-06), 'i_RBR': (95, 2.5432285268857402e-06), 'b_RBR': (96, 2.5432285268857402e-06), 'i_MD': (97, 2.5432285268857402e-06), 'b_MD': (98, 2.5432285268857402e-06)}, 'shape': {'xxx': (0, 0.14750979778789983), 'xxxx': (1, 0.4737780422735446), ',': (2, 0.010083901109101961), '.': (3, 0.07033807136807892), 'xx': (4, 0.1325505275927579), 'x': (5, 0.12742592211108314), '##xx': (6, 0.008051861516120255), '##x': (7, 0.006538640542623238), '##xxxx': (8, 0.008629174391723317), '##xxx': (9, 0.008486753594217716), "\'": (10, 0.0012563548922815558), '-': (11, 0.003428272054241978), 'dd': (12, 0.00017548276835511608), '\"': (13, 0.0007121039875280073), 'd': (14, 0.0005569670473879771), '(': (15, 5.340779906460055e-05), ')': (16, 5.086457053771481e-05), '&': (17, 4.0691656430171844e-05), ';': (18, 0.00010427236960231535), '!': (19, 5.5951027591486286e-05), '#': (20, 2.5432285268857405e-05), '##d': (21, 1.5259371161314442e-05), '?': (22, 1.7802599688200184e-05), 'dxx': (23, 1.0172914107542961e-05), ':': (24, 7.629685580657221e-06), 'dddd': (25, 4.0691656430171844e-05), 'ddd': (26, 2.0345828215085922e-05), 'ddxx': (27, 7.629685580657221e-06), '`': (28, 5.0864570537714805e-06), '%': (29, 5.0864570537714805e-06), '$': (30, 2.5432285268857402e-06), 'xxd': (31, 2.5432285268857402e-06), 'dx': (32, 2.5432285268857402e-06), 'ddxxx': (33, 2.5432285268857402e-06), 'ddx': (34, 2.5432285268857402e-06), 'ddddx': (35, 2.5432285268857402e-06), '=': (36, 2.5432285268857402e-06)}, 'ent_type': {'CARDINAL': (0, 0.017255805554919748), 'NONE': (1, 0.9697101482447908), 'DATE': (2, 0.0022380411036594513), 'FAC': (3, 0.0004450649922050046), 'PERSON': (4, 0.0012970465487117276), 'NORP': (5, 0.0028916508350690867), 'WORK_OF_ART': (6, 0.00031281710880694605), 'ORG': (7, 0.0027110816096601992), 'TIME': (8, 0.000656152959936521), 'GPE': (9, 0.001312305919873042), 'QUANTITY': (10, 0.00011444528370985832), 'PRODUCT': (11, 0.0003967436501941755), 'LOC': (12, 0.00024160671005414534), 'EVENT': (13, 0.00012970465487117277), 'ORDINAL': (14, 0.0001576801686669159), 'MONEY': (15, 7.629685580657222e-05), 'LANGUAGE': (16, 1.7802599688200184e-05), 'LAW': (17, 2.5432285268857405e-05), 'PERCENT': (18, 1.0172914107542961e-05)}, 'ent_iob': {'B': (0, 0.026327501710321183), 'O': (1, 0.9697101482447908), 'I': (2, 0.003962350044887984)}, 'sense': {'01': (0, 0.4245029895651334), 'none': (1, 0.23891088781564646), '02': (2, 0.12699103003298567), '03': (3, 0.06581366781874919), '07': (4, 0.012283793784858126), '08': (5, 0.016968420731381658), '06': (6, 0.018100157425845816), '13': (7, 0.0019175943092718483), '05': (8, 0.02107573480230213), '04': (9, 0.03430052314210798), '10': (10, 0.007749217321420851), '09': (11, 0.012774636890547074), '15': (12, 0.0011597122082598976), '27': (13, 0.0005696831900224059), '19': (14, 0.0006027451608719205), '14': (15, 0.0012334658355395841), '11': (16, 0.0026602170391224846), '12': (17, 0.005757869384869317), '33': (18, 0.0015132209734970154), '16': (19, 0.000508645705377148), '35': (20, 0.0007324498157430932), '28': (21, 0.00015005048308625867), '29': (22, 0.00039928687872106125), '20': (23, 0.0001576801686669159), '17': (24, 0.00033824939407580345), '48': (25, 0.00012207496929051554), '18': (26, 0.0003280764799682605), '42': (27, 0.0002390634815272596), '23': (28, 9.664268402165814e-05), '24': (29, 0.0003585952222908894), '39': (30, 0.00014750725455937294), '34': (31, 0.0004908431056889479), '43': (32, 0.0002263473388928309), '22': (33, 0.00028484159501120294), '30': (34, 0.00018819891098954478), '38': (35, 4.832134201082907e-05), '26': (36, 7.884008433345795e-05), '31': (37, 1.7802599688200184e-05), '32': (38, 3.0518742322628884e-05), '25': (39, 5.086457053771481e-05), '37': (40, 2.7975513795743143e-05), '46': (41, 4.3234884957057585e-05), '21': (42, 2.5432285268857405e-05), '41': (43, 2.2889056741971663e-05)}, 'sentiment': {'none': (0, 1.0)}}

        iwlst_linguistic_vocab = {'tag': {'s_EX': (40, 0.0028173198816818087), 'i_LS': (107, 5.061976195506726e-06), 'i_RB': (47, 0.004864118952038876), 's_HYPH': (39, 0.0033254982745263752), 'b_-LRB-': (110, 4.4017184308754135e-07), 'i_:': (35, 0.004795892316360307), 'i_JJ': (52, 0.0046530565532784), 's_``': (4, 0.0038116680752165646), 's_CC': (13, 0.03539509824635538), 'i_WP': (103, 1.7606873723501654e-06), 'b_NNPS': (57, 0.00020291921966335657), 'b_RBR': (121, 3.0812029016127895e-06), 's_VBN': (25, 0.012753538981618424), 'b_UH': (82, 7.614972885414466e-05), 'i_AFX': (132, 4.4017184308754135e-07), 's_POS': (83, 0.00013909430241566308), 'i_POS': (45, 0.0013581502218466088), 's_WP$': (89, 5.2380449327417426e-05), 'b_NN': (42, 0.005195348263962251), 'b_WP': (104, 1.320515529262624e-06), 's_CD': (30, 0.01051196387069512), 'b_LS': (108, 2.4209451369814776e-06), 'i_DT': (105, 3.0812029016127895e-06), 'i_-LRB-': (109, 6.60257764631312e-07), 'i_NNPS': (56, 0.0002907335023593211), 's_MD': (50, 0.010763742164941192), 'i_IN': (98, 0.0001463571378266075), 's_$': (80, 4.797873089654201e-05), 'i_VBZ': (14, 0.009155354250299317), 'i_PDT': (136, 4.4017184308754135e-07), 'i_RP': (134, 8.803436861750827e-07), 'i_,': (115, 1.0784210155644764e-05), 's_XX': (0, 0.0015045073596732163), 's_JJS': (63, 0.0013396630044369322), 'b_RBS': (123, 8.803436861750827e-07), 'b_VB': (44, 0.0026590781040918376), 's_IN': (21, 0.09271097436439187), 's_RB': (10, 0.05084292907951264), 'i_MD': (32, 0.0013585903936896965), 'b_VBG': (55, 0.0010227392774139023), 'b_NFP': (73, 4.511761391647299e-05), 's_-RRB-': (96, 0.00020710085217268823), 's_RP': (31, 0.004982965349672512), 's_,': (11, 0.054767721318402705), 'b_VBZ': (15, 0.008966080357771675), 's_VBG': (38, 0.015533444256637791), 'b_RB': (48, 0.0045447742798788646), 's_TO': (19, 0.015936421578984436), 's_AFX': (88, 6.60257764631312e-07), 'i_CC': (111, 6.822663567856891e-06), 's_NNS': (26, 0.03825291393760124), 'i_FW': (67, 3.4113317839284457e-05), 'i_VBD': (76, 0.000390652510740193), 'b_:': (36, 0.004602656877244877), 'i_RBS': (122, 8.803436861750827e-07), 's_NNPS': (49, 0.0005922512148742869), 'i_CD': (58, 0.002462981547996338), 'i_XX': (86, 1.1444467920276076e-05), 'b_JJS': (79, 0.00011818613986900486), 'b_JJ': (53, 0.0027200419043594617), 'b_IN': (99, 4.467744207338545e-05), 'i_$': (138, 2.2008592154377068e-07), 's_WRB': (61, 0.006722524473554476), 's_LS': (97, 1.892738925276428e-05), "s_\''": (7, 0.003907185365166561), 's_VBP': (8, 0.026991777589971126), 's_VBD': (28, 0.024167414958799916), 's_VBZ': (37, 0.020422432917811113), 'b_NNS': (6, 0.0040779720402845275), 'b_RP': (135, 4.4017184308754135e-07), 's_UH': (34, 0.0029429889428833015), 's_WDT': (62, 0.007128362912881189), 's_DT': (16, 0.08497627473765758), 'b_VBN': (65, 0.0006395696880061976), 's_PDT': (74, 0.0012635132755827875), 's_PRP$': (29, 0.011026304669342912), 'b_WDT': (125, 8.803436861750827e-07), 'i_VBG': (54, 0.0015412617085710262), 'b_ADD': (101, 1.1664553841819847e-05), 'b_``': (114, 1.5626100429607717e-05), 'b_MD': (33, 0.001357930135925065), 'b_XX': (87, 4.841890273962955e-06), 's_RBR': (60, 0.0015533664342559335), 'b_AFX': (133, 2.2008592154377068e-07), 'i_HYPH': (130, 1.7606873723501654e-06), 'b_HYPH': (131, 1.320515529262624e-06), 'i_WRB': (126, 2.2008592154377068e-07), 'i__SP': (92, 3.213254454539052e-05), 's_.': (12, 0.04959152052961476), 'i_NN': (41, 0.008107305091907881), 's_NN': (18, 0.10234655609549968), 'b_DT': (106, 1.9807732938939363e-06), 'i_PRP': (69, 0.00040055637720966264), 's_WP': (27, 0.006562962180435242), "i_\''": (117, 4.181632509331643e-06), 'i_NNS': (5, 0.00657044510176773), 'b_CD': (59, 0.0014312187477991407), 'i_VB': (43, 0.0032273399535178533), 's_JJ': (17, 0.04591938692865695), 'i_.': (84, 0.00017452813578421015), 's_NFP': (94, 2.641031058525248e-06), 's_JJR': (51, 0.0024739858440735265), "b_\''": (118, 3.7414606662441017e-06), 'b_.': (85, 6.888689344320023e-05), 'i_ADD': (100, 3.807486442707233e-05), 'b_PRP': (70, 0.00039285336995563067), 'b_PDT': (137, 4.4017184308754135e-07), 's_VB': (20, 0.0368428234382703), 's_SYM': (102, 4.2696668779491516e-05), 'b_WRB': (127, 2.2008592154377068e-07), 's_ADD': (119, 2.861116980069019e-06), 'b_VBD': (77, 0.0002718061131065568), 'i_VBP': (23, 0.005093668568209028), 'i_``': (113, 1.8267131488132966e-05), 'b_POS': (46, 0.0013539685893372772), 'b_-RRB-': (129, 4.4017184308754135e-07), 's_FW': (66, 0.00012412845975068667), 'i_UH': (81, 8.693393900978942e-05), 'b_NNP': (3, 0.00553031903655187), 'b_FW': (68, 1.7826959645045427e-05), 'b__SP': (93, 1.8047045566589197e-05), 'b_VBP': (24, 0.004931245158109726), 'i_WDT': (124, 1.1004296077188534e-06), 'i_VBN': (64, 0.000982683639692936), 's_PRP': (9, 0.07050760616944855), 's_RBS': (71, 0.0005838879498556237), 'i_JJS': (78, 0.00013513275582787522), 'b_,': (116, 9.90386646946968e-06), 'b_CC': (112, 2.861116980069019e-06), 's__SP': (75, 0.00019433586872314953), 'b_JJR': (91, 6.250440171843087e-05), 'i_JJR': (90, 7.108775265863794e-05), 's_:': (22, 0.003198068525952532), 'i_NNP': (2, 0.009499788717515318), 'i_RBR': (120, 3.7414606662441017e-06), 'i_NFP': (72, 7.835058806958236e-05), 's_-LRB-': (95, 0.00020335939150644411), 's_NNP': (1, 0.01795570990914853), 'b_$': (139, 2.2008592154377068e-07), 'i_-RRB-': (128, 6.60257764631312e-07)}, 'ent_type': {'LANGUAGE': (18, 0.00013205155292626242), 'FAC': (9, 0.0007460912740333826), 'LOC': (10, 0.002008063948165364), 'TIME': (4, 0.0022215472920628213), 'QUANTITY': (8, 0.0016713324882033946), 'DATE': (5, 0.012843994295372913), 'ORDINAL': (11, 0.001517272343122755), 'NORP': (12, 0.0019468800619761955), 'PERCENT': (13, 0.0015075885625748292), 'ORG': (1, 0.007963368899218256), 'CARDINAL': (6, 0.008657960067610396), 'MONEY': (15, 0.0013748767518839355), 'WORK_OF_ART': (14, 0.0021955771533206563), 'EVENT': (16, 0.00040869955630678216), 'PRODUCT': (2, 0.00044919536587083596), 'PERSON': (3, 0.008482771674061553), 'NONE': (0, 0.939551200788788), 'LAW': (17, 0.00017452813578421015), 'GPE': (7, 0.006146999788717515)}, 'shape': {'##d': (23, 0.0004241055708148461), ']': (32, 0.0002020388759771815), '[XXX]': (42, 2.641031058525248e-06), '##|': (54, 1.1004296077188534e-06), '[': (31, 0.0002020388759771815), '##(c)': (33, 2.883125572223396e-05), "##\'": (64, 4.4017184308754135e-07), '## 1/2 ': (63, 2.2008592154377068e-07), '#': (47, 4.621804352419185e-06), 'd': (5, 0.0007062557222339602), ',': (9, 0.055332681879005564), '##dd': (34, 1.892738925276428e-05), 'xxxxdd': (61, 2.2008592154377068e-07), 'dx': (39, 3.279280231002183e-05), '<': (0, 0.0007504929924642581), '## 1/4 ': (55, 1.7606873723501654e-06), 'ddddx': (30, 9.815832100852173e-05), 'SS': (60, 5.5021480385942675e-06), 'dxx': (48, 8.583350940207057e-06), '##ddd': (49, 1.1004296077188534e-06), '##+-': (41, 1.1004296077188534e-06), '.': (10, 0.04682812169871118), '_': (66, 5.5021480385942675e-06), 'xx': (8, 0.15714244841185998), 'x': (12, 0.05323482287485034), '##!': (65, 2.2008592154377068e-07), '##xxx': (16, 0.009680699345024298), '(': (37, 1.5406014508063947e-06), ':': (22, 0.002457039228114656), 'xxx': (1, 0.17065242270582434), '##C/': (35, 1.1004296077188534e-06), '+': (38, 7.262835410944433e-06), 'dddxx': (50, 8.803436861750827e-07), '##x': (21, 0.0056542274103810125), '=': (3, 0.0007551147968166772), '##"': (57, 4.621804352419185e-06), '##dx': (40, 1.320515529262624e-06), '@': (51, 1.9807732938939363e-06), 'ddx': (28, 8.451299387280794e-05), '$': (25, 4.665821536727939e-05), '##$?': (56, 6.60257764631312e-07), '-': (15, 0.013175223607296288), '/': (24, 0.000441492358616804), 'ddxx': (27, 0.00016418409747165293), 'P': (52, 4.4017184308754135e-06), '<<': (45, 0.0008653778435101063), 'xxxx': (6, 0.42272607225860975), '>': (7, 0.0007504929924642581), "\'": (11, 0.022324415451792378), '!': (14, 0.0003395925769420382), '##xxxx': (18, 0.008360623987604761), 'ddd': (20, 0.001594962673427706), ';': (13, 0.0007110976125079231), '##PS': (59, 5.282062117050496e-06), 'xd': (44, 3.521374744700331e-06), 'dd': (17, 0.002954213324882034), '##xx': (2, 0.008642774139023875), '"': (4, 0.0077545073596732165), 'dddd': (26, 0.000707356151841679), '^': (58, 1.5406014508063947e-06), '?': (19, 0.004045839495739136), ')': (36, 5.282062117050496e-06), '%': (46, 5.9423198816818084e-06), 'xxd': (43, 2.861116980069019e-06), '##xd': (62, 2.2008592154377068e-07), '&': (29, 2.7290654271427565e-05), '*': (53, 6.60257764631312e-07)}, 'ent_iob': {'O': (0, 0.939551200788788), 'I': (2, 0.02343805021480386), 'B': (1, 0.037010748996408195)}, 'pos': {'i_ADV': (32, 0.001064335516585675), 's_CCONJ': (10, 0.03539509824635538), 'i_CCONJ': (48, 6.822663567856891e-06), 's_SPACE': (38, 0.00019433586872314953), 'i_NOUN': (5, 0.01467775019367561), 's_INTJ': (23, 0.0029429889428833015), 'b_NOUN': (6, 0.009273320304246779), 'b_SPACE': (43, 1.8047045566589197e-05), 's_VERB': (7, 0.10097255968730193), 's_NOUN': (15, 0.1375160662722727), 'b_DET': (47, 3.3012888231565603e-06), 'b_SYM': (51, 2.2008592154377068e-07), 's_ADJ': (14, 0.04973369603493204), 's_PRON': (8, 0.08297129199239384), 's_PART': (16, 0.019603053031903656), 'i_PUNCT': (24, 0.0050850852172688215), 'i_INTJ': (40, 8.693393900978942e-05), 'i_ADP': (44, 0.0001472374815127826), 'i_DET': (46, 4.621804352419185e-06), 's_PROPN': (1, 0.018547961124022818), 'b_PRON': (37, 0.0003941738854848933), 's_ADV': (9, 0.05617517078667512), 'i_PART': (26, 0.0051627755475737724), 'b_ADV': (33, 0.0007564353123459399), 's_SYM': (39, 9.067539967603353e-05), 's_ADP': (18, 0.0782326220156349), 'i_SPACE': (42, 3.213254454539052e-05), 'b_VERB': (22, 0.005069018944996126), 'b_PROPN': (3, 0.005733238256215226), 'i_AUX': (11, 0.01478515212338897), 'b_CCONJ': (49, 2.861116980069019e-06), 'b_AUX': (12, 0.014779429889428833), 's_X': (0, 0.0016504243256567363), 'b_INTJ': (41, 7.614972885414466e-05), 'b_X': (35, 3.6754348897809705e-05), 'b_PUNCT': (25, 0.004748133671385309), 'b_NUM': (31, 0.0014312187477991407), 'b_ADJ': (29, 0.0029009525318684413), 'i_ADJ': (28, 0.0048597172336080005), 's_AUX': (17, 0.04650261462074794), 's_NUM': (20, 0.01051196387069512), 'i_SYM': (50, 2.2008592154377068e-07), 'i_X': (34, 8.869462638213959e-05), 'i_PROPN': (2, 0.009790522219874639), 's_DET': (13, 0.1044468360447919), 's_PUNCT': (4, 0.11901476336361716), 'i_VERB': (21, 0.006964398901331079), 'i_PRON': (36, 0.0004023170645820128), 'b_PART': (27, 0.0051464891893795335), 'i_NUM': (30, 0.002462981547996338), 's_SCONJ': (19, 0.019461317698429466), 'b_ADP': (45, 4.511761391647299e-05)}, 'sentiment': {'none': (0, 1.0)}, 'sense': {'11': (10, 0.00356627227269526), '30': (32, 2.883125572223396e-05), '40': (41, 1.848721740967674e-05), '27': (35, 0.00023329107683639692), '23': (21, 0.00015736143390379604), '25': (34, 5.128001971969857e-05), '15': (14, 0.0005746443411507853), '46': (17, 0.0002951352207901965), '13': (18, 0.0008986108176632158), '07': (7, 0.009269798929502077), '19': (15, 0.0007300250017606873), '22': (29, 0.0004883706599056271), '21': (30, 0.00021656454679907036), '49': (40, 1.1004296077188535e-05), '55': (47, 4.181632509331643e-06), '24': (31, 0.0002528787238537925), '01': (1, 0.30214891893795337), '33': (33, 7.306852595253186e-05), '32': (36, 5.414113669976759e-05), '37': (48, 1.2324811606451158e-05), 'none': (0, 0.4468496901190225), '41': (42, 5.788259736601169e-05), '43': (20, 0.00016726530037326572), '52': (49, 8.803436861750827e-07), '18': (28, 0.0008400679625325727), '38': (19, 0.00022206669483766463), '29': (39, 0.00014151524755264455), '34': (16, 0.0007436703288964012), '04': (6, 0.025558357982956546), '06': (9, 0.012744735544756672), '10': (13, 0.004586370519050638), '02': (2, 0.09163453412212128), '03': (4, 0.05431280371857173), '26': (23, 0.00012126734277061765), '45': (50, 3.7414606662441017e-06), '42': (24, 0.00033981266286358194), '35': (22, 0.001539501021198676), '36': (45, 4.4017184308754135e-06), '48': (38, 5.48013944643989e-05), '16': (37, 0.00023989365448271005), '50': (52, 2.2008592154377068e-07), '31': (46, 7.042749489400662e-06), '17': (27, 0.0005279861257835058), '47': (51, 6.60257764631312e-07), '59': (54, 2.2008592154377068e-07), '14': (5, 0.0005398707655468695), '28': (26, 0.00015956229311923375), '12': (12, 0.005558490034509473), '51': (53, 2.2008592154377068e-07), '44': (44, 3.0812029016127895e-06), '05': (3, 0.01822047327276569), '20': (25, 0.0003461951545883513), '09': (11, 0.005882236425100359), '08': (8, 0.009470077118106908), '39': (43, 3.521374744700331e-05)}}

        # TODO after being done with testing, you'd need to call extract_linguistic_vocabs() to extract the vocab
        # 1 ../../.data/iwslt/de-en/train.de-en.en
        # 1 ../../.data/multi30k/train.en
        features_list = ['pos', 'shape', 'tag']
        dataset_address = sys.argv[2]
        if "iwslt" in dataset_address:
            ling_vocab =iwlst_linguistic_vocab
        elif "multi30k" in dataset_address:
            ling_vocab = multi30k_linguistic_vocab
        else:
            raise ValueError("For new datasets you need to train a new linguistic vocabulary")
        reverse_linguistic_vocab = create_empty_linguistic_vocab()
        for key in ling_vocab:
            for key2 in ling_vocab[key]:
                reverse_linguistic_vocab[key][ling_vocab[key][key2][0]] = key2
        project_sub_layers_trainer(dataset_address, bert_tknizer, ling_vocab, features_list)
        project_sub_layers_tester(["A little girl climbing into a wooden playhouse."],
                                  bert_tknizer, ling_vocab, features_list)

