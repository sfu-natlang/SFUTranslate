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
# model_name = 'bert-base-uncased'
model_name = 'bert-base-german-dbmdz-uncased'
language_extension = ".de"
number_of_bert_layers = 13
D_in, H, D_out = 768, 1024, 768
epochs = 3
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
        torch.save({'model': model}, save_model_name+".module")
        torch.save({'features_list': features_list, 'softmax': nn.Softmax(dim=-1), 'head_converters': model.encoders,
                    'bert_weights': model.bert_weights_for_average_pooling}, save_model_name)


def project_sub_layers_tester(input_sentences, bert_tokenizer, linguistic_vocab, required_features_list,
                              load_model_name="project_sublayers.pt"):
    bert_lm = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    saved_obj = torch.load(load_model_name+".module", map_location=lambda storage, loc: storage)
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
        # TODO after being done with testing, you'd need to call extract_linguistic_vocabs() to extract the vocab
        # 1 ../../.data/iwslt/de-en/train.de-en.en
        # 1 ../../.data/multi30k/train.en
        features_list = ['pos', 'shape', 'tag']
        dataset_address = sys.argv[2]
        if "iwslt" in dataset_address:
            # ling_vocab =iwlst_linguistic_vocab
            smn = "iwslt_head_conv" + language_extension
        elif "multi30k" in dataset_address:
            # ling_vocab = multi30k_linguistic_vocab
            smn = "multi30k_head_conv" + language_extension
        else:
            raise ValueError("For new datasets you need to train a new linguistic vocabulary")
        ling_vocab = extract_linguistic_vocabs(sys.argv[2], bert_tknizer)
        reverse_linguistic_vocab = create_empty_linguistic_vocab()
        for key in ling_vocab:
            for key2 in ling_vocab[key]:
                reverse_linguistic_vocab[key][ling_vocab[key][key2][0]] = key2
        project_sub_layers_trainer(dataset_address, bert_tknizer, ling_vocab, features_list, save_model_name=smn)
        project_sub_layers_tester(["A little girl climbing into a wooden playhouse."],
                                  bert_tknizer, ling_vocab, features_list, load_model_name=smn)

