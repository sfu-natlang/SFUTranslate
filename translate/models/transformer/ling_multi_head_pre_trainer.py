"""
The test script for training an intermediary sub-sectioned layer which does contain the same exact information as bert

"""
import sys
import spacy
from spacy.tokenizer import Tokenizer
from torch import optim
from torch import nn
import torch.nn.init as init
import torch
from pathlib import Path
from tqdm import tqdm
from textblob import TextBlob
from nltk.wsd import lesk
import unidecode

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from transformers import BertTokenizer, BertForMaskedLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ###############################################CONFIGURATIONS########################################################
model_name = 'bert-base-uncased'
number_of_bert_layers = 13
D_in, H, D_out = 768, 1024, 768
epochs = 10
lr = 0.01
batch_size = 32
max_norm = 5

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
            # print(spacy_doc)
            # print(bert_doc)
            # print(bert_f_pointer)
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


def extract_linguistic_vocabs(file_adr, bert_tokenizer):
    vocabs = {"pos": {}, "tag": {}, "shape": {}, "ent_type": {}, "ent_iob": {}, "sense": {}, "sentiment": {}}
    itr = tqdm(get_next_batch(file_adr, batch_size))
    for input_sentences in itr:
        for sent in input_sentences:
            res = extract_linguistic_features(sent, bert_tokenizer)
            for res_item in res:
                for key in res_item:
                    value = res_item[key]
                    if key in vocabs and value not in vocabs[key]:
                        vocabs[key][value] = len(vocabs[key])
    return vocabs


class SubLayerED(torch.nn.Module):
    def __init__(self, D_in, Hs, D_out, feature_sizes, padding_index=0):
        super(SubLayerED, self).__init__()
        self.equal_length_Hs = (sum([Hs[0] == h for h in Hs]) == len(Hs))
        self.consider_adversarial_loss = False
        self.encoders = nn.ModuleList([nn.Linear(D_in, h) for h in Hs])
        self.decoder = nn.Linear(sum(Hs), D_out)
        self.feature_classifiers = nn.ModuleList([nn.Linear(h, o) for h, o in zip(Hs[:-1], feature_sizes)])

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.class_loss_fn = nn.CrossEntropyLoss(ignore_index=padding_index, reduction='sum')
        self.pair_distance = nn.PairwiseDistance(p=2)
        self.discriminator = nn.Linear(D_out, 1)

        self.bert_weights_for_average_pooling = nn.Parameter(torch.zeros(number_of_bert_layers), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, features):
        encoded = [self.encoders[i](x) for i in range(len(self.encoders))]
        y_pred = self.decoder(torch.cat(encoded, 1))
        ling_classes = [self.feature_classifiers[i](encoded[i]) for i in range(len(self.encoders)-1)]
        loss = self.loss_fn(y_pred, x)
        feature_pred_correct = [(lc.argmax(dim=-1) == features[ind]) for ind, lc in enumerate(ling_classes)]
        for ind, lc in enumerate(ling_classes):
            loss += self.class_loss_fn(lc, features[ind])
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
    for sent in input_sentences:
        sent_extracted_features = [[padding_value] for _ in required_features_list]
        res = extract_linguistic_features(sent, bert_tokenizer)
        for token_feature in res:
            for ind, elem in enumerate(required_features_list):
                assert elem in token_feature, "feature {} is required to be extracted!"
                feature = token_feature[elem]
                feature_id = linguistic_vocab[elem][feature] + 1  # the first index is <PAD>
                sent_extracted_features[ind].append(feature_id)
        for ind in range(len(required_features_list)):
            sent_extracted_features[ind].append(padding_value)
            extracted_features[ind].append(torch.tensor(sent_extracted_features[ind], device=device).long())

    return [torch.nn.utils.rnn.pad_sequence(extracted_features[ind], batch_first=True, padding_value=padding_value)
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
    opt = optim.SGD(model.parameters(), lr=float(lr))
    for t in range(epochs):
        all_loss = 0.0
        all_tokens_count = 0.0
        itr = tqdm(get_next_batch(file_adr, batch_size))
        feature_pred_corrects = [0 for _ in range(len(required_features_list))]
        feature_pred_correct_all = 0.0
        for input_sentences in itr:
            sequences = [torch.tensor(bert_tokenizer.encode(input_sentence, add_special_tokens=True), device=device)
                         for input_sentence in input_sentences]
            features = map_sentences_to_vocab_ids(
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
                permitted_to_continue = True
                for f in features:
                    if s < f.size(1):
                        features_selected.append(f.select(1, s))
                    else:
                        permitted_to_continue = False
                if not permitted_to_continue:
                    continue
                _, loss, feature_pred_correct = model(x, features_selected)
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
        features = map_sentences_to_vocab_ids(input_sentences, required_features_list, linguistic_vocab, bert_tokenizer)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=bert_tokenizer.pad_token_id)
        outputs = bert_lm(input_ids, masked_lm_labels=input_ids)[2]  # (batch_size * [input_length + 2] * 768)
        all_layers_embedded = torch.cat([o.detach().unsqueeze(0) for o in outputs], dim=0)
        embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0),
                                model.softmax(model.bert_weights_for_average_pooling))
        for s in range(1, embedded.size(1)-1):
            x = embedded.select(1, s)
            features_selected = [f.select(1, s) for f in features]
            _, loss, feature_pred_correct = model(x, features_selected)
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
        print(extract_linguistic_vocabs(sys.argv[2], bert_tknizer))
    elif running_mode == 2:
        multi30k_linguistic_vocab = {'pos': {'s_NUM': 0, 's_ADJ': 1, 's_PUNCT': 2, 's_NOUN': 3, 's_AUX': 4, 's_ADV': 5, 's_SCONJ': 6, 's_ADP': 7, 's_VERB': 8, 's_DET': 9, 'i_NOUN': 10, 'b_NOUN': 11, 'i_ADJ': 12, 'b_ADJ': 13, 's_PART': 14, 's_CCONJ': 15, 'i_VERB': 16, 'b_VERB': 17, 's_PRON': 18, 's_PROPN': 19, 'i_PROPN': 20, 'b_PROPN': 21, 'i_PART': 22, 'b_PART': 23, 'i_AUX': 24, 'b_AUX': 25, 'i_ADV': 26, 'b_ADV': 27, 's_INTJ': 28, 'i_DET': 29, 'b_DET': 30, 's_X': 31, 'i_NUM': 32, 'b_NUM': 33, 's_SYM': 34, 'i_PRON': 35, 'b_PRON': 36, 'i_ADP': 37, 'b_ADP': 38, 'i_PUNCT': 39, 'b_PUNCT': 40, 'i_X': 41, 'b_X': 42, 'i_SPACE': 43, 'b_SPACE': 44}, 'tag': {'s_CD': 0, 's_JJ': 1, 's_,': 2, 's_NNS': 3, 's_VBP': 4, 's_RB': 5, 's_IN': 6, 's_.': 7, 's_VBG': 8, 's_DT': 9, 'i_NN': 10, 'b_NN': 11, 's_NN': 12, 's_VBZ': 13, 's_PRP$': 14, 's_VBN': 15, 'i_JJ': 16, 'b_JJ': 17, 's_CC': 18, 'i_VBG': 19, 'b_VBG': 20, 's_TO': 21, 's_WDT': 22, 'i_NNS': 23, 'b_NNS': 24, 's_PRP': 25, 's_RP': 26, 's_NNP': 27, 'i_NNP': 28, 'b_NNP': 29, 'i_VBN': 30, 'b_VBN': 31, 's_VB': 32, 'i_POS': 33, 'b_POS': 34, 'i_VBZ': 35, 'b_VBZ': 36, 's_WP': 37, 's_HYPH': 38, 's_JJR': 39, 's_VBD': 40, 's_EX': 41, 'i_RB': 42, 'b_RB': 43, 'i_JJR': 44, 'b_JJR': 45, 's_``': 46, 'i_NNPS': 47, 'b_NNPS': 48, "s_''": 49, 's_WRB': 50, 's_MD': 51, 'i_VBP': 52, 'b_VBP': 53, 's_-LRB-': 54, 's_-RRB-': 55, 's_:': 56, 'i_VB': 57, 'b_VB': 58, 's_PDT': 59, 'i_VBD': 60, 'b_VBD': 61, 's_NNPS': 62, 's_UH': 63, 's_POS': 64, 'i_DT': 65, 'b_DT': 66, 's_RBS': 67, 's_WP$': 68, 's_FW': 69, 'i_CD': 70, 'b_CD': 71, 's_$': 72, 'i_PRP': 73, 'b_PRP': 74, 's_JJS': 75, 's_RBR': 76, 's_LS': 77, 'i_IN': 78, 'b_IN': 79, 'i_.': 80, 'b_.': 81, 's_XX': 82, 'i_XX': 83, 'b_XX': 84, 'i_FW': 85, 'b_FW': 86, 'i_JJS': 87, 'b_JJS': 88, 'i_:': 89, 'b_:': 90, 'i__SP': 91, 'b__SP': 92, 'i_RBS': 93, 'b_RBS': 94, 'i_RBR': 95, 'b_RBR': 96, 'i_MD': 97, 'b_MD': 98}, 'shape': {'xxx': 0, 'xxxx': 1, ',': 2, '.': 3, 'xx': 4, 'x': 5, '##xx': 6, '##x': 7, '##xxxx': 8, '##xxx': 9, "'": 10, '-': 11, 'dd': 12, '"': 13, 'd': 14, '(': 15, ')': 16, '&': 17, ';': 18, '!': 19, '#': 20, '##d': 21, '?': 22, 'dxx': 23, ':': 24, 'dddd': 25, 'ddd': 26, 'ddxx': 27, '`': 28, '%': 29, '$': 30, 'xxd': 31, 'dx': 32, 'ddxxx': 33, 'ddx': 34, 'ddddx': 35, '=': 36}, 'ent_type': {'CARDINAL': 0, 'NONE': 1, 'DATE': 2, 'FAC': 3, 'PERSON': 4, 'NORP': 5, 'WORK_OF_ART': 6, 'ORG': 7, 'TIME': 8, 'GPE': 9, 'QUANTITY': 10, 'PRODUCT': 11, 'LOC': 12, 'EVENT': 13, 'ORDINAL': 14, 'MONEY': 15, 'LANGUAGE': 16, 'LAW': 17, 'PERCENT': 18}, 'ent_iob': {'B': 0, 'O': 1, 'I': 2}, 'sense': {'01': 0, 'none': 1, '02': 2, '03': 3, '07': 4, '08': 5, '06': 6, '13': 7, '05': 8, '04': 9, '10': 10, '09': 11, '15': 12, '27': 13, '19': 14, '14': 15, '11': 16, '12': 17, '33': 18, '16': 19, '35': 20, '28': 21, '29': 22, '20': 23, '17': 24, '48': 25, '18': 26, '42': 27, '23': 28, '24': 29, '39': 30, '34': 31, '43': 32, '22': 33, '30': 34, '38': 35, '26': 36, '31': 37, '32': 38, '25': 39, '37': 40, '46': 41, '21': 42, '41': 43}, 'sentiment': {'none': 0}}

        iwlst_linguistic_vocab = {'tag': {'s_CC': 13, 's_-LRB-': 95, 's_RP': 31, 'b_LS': 108, 'b__SP': 93, 'b_VBP': 24, 'i_DT': 105, 'i_NFP': 72, 'b_RP': 135, 'i_JJR': 90, 'b_.': 85, 'b_NFP': 73, 's_PDT': 74, 'b_RB': 48, 's_VBP': 8, 'i_UH': 81, 's_RB': 10, 'i_VBP': 23, "s_''": 7, 's_DT': 16, 'i__SP': 92, 'i_NNP': 2, 's_HYPH': 39, 'b_NN': 42, 'i_LS': 107, 'i_PRP': 69, 's_NN': 18, 'i_VB': 43, 'b_VBN': 65, 'b_DT': 106, 's_$': 80, 'i_JJ': 52, 's_JJ': 17, 'i_POS': 45, 's_FW': 66, 'i_WP': 103, 's_VBZ': 37, 'i_NNPS': 56, 's__SP': 75, 'i_VBD': 76, 's_AFX': 88, 's_VBG': 38, "b_''": 118, 's_VBD': 28, 's_WDT': 62, 'b_JJS': 79, 'i_``': 113, 'b_XX': 87, 'b_UH': 82, 'b_$': 139, 's_LS': 97, 's_PRP$': 29, 'b_JJ': 53, 's_JJR': 51, 's_RBS': 71, 's_RBR': 60, 'i_RP': 134, 'i_,': 115, 'b_FW': 68, 'i_AFX': 132, 's_XX': 0, 'b_MD': 33, 's_EX': 40, 'i_RB': 47, 's_NFP': 94, 's_-RRB-': 96, 'i_WDT': 124, 'b_NNS': 6, 'b_VBZ': 15, 'i_JJS': 78, 'i_ADD': 100, 's_,': 11, 'b_WDT': 125, 's_TO': 19, 's_SYM': 102, 'i_MD': 32, 'i_RBR': 120, 's_WP': 27, 's_POS': 83, 'b_NNP': 3, 'b_NNPS': 57, 'i_NNS': 5, 'b_AFX': 133, 's_UH': 34, 's_NNPS': 49, 'b_WP': 104, 'b_PRP': 70, 'i_PDT': 136, 'i_:': 35, 'i_-LRB-': 109, 'i_$': 138, "i_''": 117, 'b_VB': 44, 'i_NN': 41, 'b_POS': 46, 's_PRP': 9, 's_WP$': 89, 'b_VBG': 55, 'i_CD': 58, 'i_IN': 98, 'b_CC': 112, 's_.': 12, 'b_CD': 59, 'b_,': 116, 'b_PDT': 137, 'i_.': 84, 'i_-RRB-': 128, 'b_:': 36, 's_WRB': 61, 'i_VBN': 64, 's_:': 22, 's_NNS': 26, 'b_WRB': 127, 'b_HYPH': 131, 's_``': 4, 'i_VBZ': 14, 's_ADD': 119, 'b_JJR': 91, 'b_VBD': 77, 's_CD': 30, 'b_``': 114, 's_MD': 50, 'b_RBS': 123, 'i_XX': 86, 's_VB': 20, 'i_VBG': 54, 'b_-LRB-': 110, 'b_ADD': 101, 'b_RBR': 121, 'i_CC': 111, 'b_IN': 99, 'i_WRB': 126, 'b_-RRB-': 129, 's_NNP': 1, 's_IN': 21, 'i_RBS': 122, 's_VBN': 25, 's_JJS': 63, 'i_HYPH': 130, 'i_FW': 67}, 'shape': {'xd': 44, 'ddxx': 27, "'": 11, 'dd': 17, '?': 19, '## 1/4 ': 55, 'dxx': 48, '%': 46, '##dx': 40, '##C/': 35, 'xxd': 43, '##(c)': 33, '.': 10, 'xx': 8, ':': 22, '##|': 54, '"': 4, '##xd': 62, "##'": 64, '=': 3, '>': 7, 'xxx': 1, '*': 53, '+': 38, 'ddd': 20, '##PS': 59, '<': 0, '## 1/2 ': 63, '@': 51, 'ddddx': 30, 'x': 12, '#': 47, '##d': 23, '&': 29, 'd': 5, '##!': 65, 'dx': 39, ';': 13, '##xxxx': 18, '##ddd': 49, 'xxxxdd': 61, '##dd': 34, '[': 31, '$': 25, '/': 24, '[XXX]': 42, '##$?': 56, '(': 37, 'dddxx': 50, 'P': 52, '##x': 21, '-': 15, 'xxxx': 6, ']': 32, 'SS': 60, '^': 58, '!': 14, 'dddd': 26, ')': 36, '##+-': 41, 'ddx': 28, '_': 66, ',': 9, '##xxx': 16, '##"': 57, '<<': 45, '##xx': 2}, 'pos': {'i_SPACE': 42, 's_NUM': 20, 'b_ADV': 33, 's_PRON': 8, 's_SYM': 39, 'i_NUM': 30, 'i_PUNCT': 24, 'b_PART': 27, 'i_AUX': 11, 's_PROPN': 1, 's_ADV': 9, 'i_X': 34, 'b_NOUN': 6, 's_DET': 13, 'i_SYM': 50, 'i_ADJ': 28, 'b_SPACE': 43, 'b_PRON': 37, 's_INTJ': 23, 'i_PART': 26, 'b_PUNCT': 25, 'i_ADP': 44, 's_CCONJ': 10, 'b_ADJ': 29, 'i_ADV': 32, 's_ADJ': 14, 'i_NOUN': 5, 'b_INTJ': 41, 'b_AUX': 12, 'i_INTJ': 40, 's_SCONJ': 19, 'b_NUM': 31, 's_X': 0, 'i_PRON': 36, 'b_PROPN': 3, 's_PART': 16, 'b_SYM': 51, 's_ADP': 18, 'b_ADP': 45, 's_PUNCT': 4, 'i_DET': 46, 'i_PROPN': 2, 'b_X': 35, 's_VERB': 7, 'i_CCONJ': 48, 's_SPACE': 38, 'b_VERB': 22, 's_AUX': 17, 'b_DET': 47, 'b_CCONJ': 49, 'i_VERB': 21, 's_NOUN': 15}, 'sense': {'29': 39, '28': 26, '03': 4, '21': 30, '10': 13, '08': 8, '13': 18, '01': 1, '59': 54, '32': 36, '19': 15, '40': 41, '31': 46, '50': 52, '11': 10, '16': 37, '43': 20, '41': 42, '26': 23, '14': 5, '33': 33, '23': 21, '12': 12, '30': 32, '17': 27, '06': 9, '27': 35, 'none': 0, '07': 7, '24': 31, '55': 47, '20': 25, '47': 51, '49': 40, '46': 17, '09': 11, '04': 6, '42': 24, '48': 38, '52': 49, '35': 22, '38': 19, '18': 28, '02': 2, '39': 43, '25': 34, '05': 3, '51': 53, '37': 48, '44': 44, '36': 45, '34': 16, '22': 29, '45': 50, '15': 14}, 'sentiment': {'none': 0}, 'ent_iob': {'O': 0, 'I': 2, 'B': 1}, 'ent_type': {'TIME': 4, 'GPE': 7, 'LAW': 17, 'QUANTITY': 8, 'EVENT': 16, 'LOC': 10, 'WORK_OF_ART': 14, 'MONEY': 15, 'DATE': 5, 'LANGUAGE': 18, 'ORDINAL': 11, 'PRODUCT': 2, 'NORP': 12, 'CARDINAL': 6, 'NONE': 0, 'PERCENT': 13, 'PERSON': 3, 'ORG': 1, 'FAC': 9}}


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
        project_sub_layers_trainer(dataset_address, bert_tknizer, ling_vocab, features_list)
        project_sub_layers_tester(["A little girl climbing into a wooden playhouse."],
                                  bert_tknizer, ling_vocab, features_list)

