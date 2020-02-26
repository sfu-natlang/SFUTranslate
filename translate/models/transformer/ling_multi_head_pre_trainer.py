"""
The test script for training an intermediary sub-sectioned layer which does contain the same exact information as bert

"""
import sys
import spacy
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
        for _ in range(fertility):
            linguistic_features["token"] = bert_doc[bert_doc_pointer]
            result.append(linguistic_features.copy())
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
                features_selected = [f.select(1, s) if s < f.size(1) else None for f in features]
                if None in features_selected:
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
    bert_tknizer = BertTokenizer.from_pretrained(model_name)
    running_mode = int(sys.argv[1])
    if running_mode == 0:
        projection_trainer(sys.argv[2], bert_tknizer)
    elif running_mode == 1:
        print(extract_linguistic_vocabs(sys.argv[2], bert_tknizer))
    elif running_mode == 2:
        multi30k_linguistic_vocab = {
            'pos': {'X': 15, 'PUNCT': 2, 'DET': 9, 'ADV': 5, 'CCONJ': 11, 'NOUN': 3, 'PROPN': 13, 'NUM': 0, 'INTJ': 14,
                    'VERB': 8, 'SYM': 16, 'PRON': 12, 'SCONJ': 6, 'AUX': 4, 'ADP': 7, 'ADJ': 1, 'PART': 10},
            'sense': {'26': 36, '10': 10, '46': 41, '37': 40, '41': 43, '38': 35, '17': 24, '28': 21, '29': 22,
                      '33': 18, '16': 19, '02': 2, '08': 5, '14': 15, '43': 32, '01': 0, '15': 12, '23': 28, '48': 25,
                      '11': 16, '24': 29, '03': 3, '04': 9, '31': 37, '34': 31, '13': 7, '09': 11, 'none': 1, '05': 8,
                      '19': 14, '18': 26, '20': 23, '22': 33, '35': 20, '21': 42, '30': 34, '07': 4, '12': 17, '39': 30,
                      '27': 13, '32': 38, '06': 6, '42': 27, '25': 39},
            'sentiment': {'positive': 1, 'none': 0, 'negative': 2},
            'ent_iob': {'B': 0, 'I': 2, 'O': 1},
            'ent_type': {'LOC': 12, 'ORG': 7, 'WORK_OF_ART': 6, 'PERCENT': 18, 'MONEY': 15, 'CARDINAL': 0,
                         'PRODUCT': 11, 'NORP': 5, 'NONE': 1, 'LAW': 17, 'DATE': 2, 'LANGUAGE': 16, 'TIME': 8,
                         'ORDINAL': 14, 'PERSON': 4, 'GPE': 9, 'EVENT': 13, 'QUANTITY': 10, 'FAC': 3},
            'shape': {'XXXx': 75, 'X&xxx;X.': 40, 'X': 8, 'XxxxxXxxx': 33, 'dxx': 45, "'xx": 23, 'd-xxxx': 53,
                      'xxxxd': 65, 'XXxxxx': 71, '.x': 56, "x'xxxx": 38, 'Xxx.': 55, 'xxxx': 1, 'X.X': 47, ':': 48,
                      'X.X.': 61, '(': 19, "'": 29, 'XxxxXxxx': 10, 'xXxx': 51, 'x': 7, 'ddx': 64, 'dx': 76, 'XXX': 13,
                      'dd:dd': 70, 'Xxxxx': 3, 'dd': 16, 'xxxx-': 59, 'XxXxxxx': 11, 'X.': 31, 'xx': 6, 'Xx.': 32,
                      'Xx': 14, 'X.X.X.X.X.': 34, 'XxXxxx': 73, 'd': 18, 'Xxxx': 9, '#': 39, 'xxx': 4, 'XxxxXxxxx': 37,
                      'XxxxxXxx': 27, 'ddd': 57, "x'x": 36, 'XxxXx': 72, '=': 79, 'XXx': 25, 'XX': 26, 'xXXXxxxx': 69,
                      'x-d': 30, 'Xxx': 0, ',': 2, 'ddddx': 78, 'XxxXxxx': 52, 'X.X.X.': 49, "'x": 12, ';': 22, '?': 44,
                      '"': 17, '%': 63, 'dd,ddd': 50, 'XxxxxXxxxx': 46, '!': 28, '&': 21, '.': 5, 'xXxxx': 42,
                      'xxxx.xxx': 58, '-': 15, 'dddd': 43, 'ddxx': 54, ')': 20, 'x&xxx;x': 67, 'xXxxxx': 66, '$': 68,
                      'X&xxx;X': 41, 'xxxXxxxx': 60, 'd.dd': 35, 'XXXX': 24, 'ddxxx': 77, 'xxd': 74, '`': 62},
            'tag': {'JJ': 1, 'JJR': 24, 'CC': 14, 'PDT': 35, 'WDT': 16, 'UH': 36, 'CD': 0, ':': 34, 'VBG': 8, 'RB': 5,
                    'VB': 20, 'VBZ': 11, 'POS': 21, 'WP': 22, 'LS': 43, 'NNS': 3, 'NN': 10, '``': 27, 'XX': 44,
                    'PRP$': 12, 'TO': 15, 'HYPH': 23, 'PRP': 17, '-LRB-': 32, 'NNP': 19, 'WP$': 38, 'WRB': 30,
                    '-RRB-': 33, 'VBD': 25, 'RBS': 37, 'FW': 39, 'IN': 6, 'JJS': 41, 'RBR': 42, 'MD': 31, '.': 7,
                    'VBP': 4, 'DT': 9, ',': 2, 'NNPS': 28, 'EX': 26, 'RP': 18, 'VBN': 13, '$': 40, "''": 29}}

        iwlst_linguistic_vocab = {
            'shape': {'': 586, 'XXdxxxXdxxxx': 479, 'ddx': 55, '_': 574, 'XX"xxxx': 541, 'ddd-xxx': 328,
                      'XxxxxXxxXxxXXxxx': 523, 'XXddd': 317, 'xxxxxx': 197, 'x': 13, "Xxx'xxxx": 170,
                      'XXdxxxxXPxxxx': 333, '<': 0, 'Xxxx-': 464, 'xx 1/2 ': 440, 'XxXxx': 133, 'xXX': 461,
                      'XxxxxX"xxx': 526, '-d': 223, 'XxXPx': 473, 'xxx': 1, 'X.x': 183, 'Xdxdxdx': 422, '.dd': 81,
                      '""Xxx': 497, 'XxxXxxx.xxx': 178, 'dd,dddxx': 519, '/xxx': 38, 'dd-xx': 169, 'XX&xxx;X': 222,
                      "x'xxx": 343, 'XxxxxXxxx': 155, 'XX': 32, 'XxxxXx': 453, 'xxx-xx-xx-xxx-xx-xxx-xx-xx-xxx.xxx': 107,
                      'dddxxx': 571, "xx'xx": 64, ' ': 40, "'X": 618, 'ddxx': 46, 'xxx-dd': 129, 'XXXX': 42, 'XdX': 227,
                      '....': 203, 'xxxx-dd': 297, 'xXxxxxx': 567, 'dd,ddd,ddd': 521, '-': 22, 'ddd': 25,
                      'XxxxXxxxxXxx': 316, 'xxxx.x': 562, 'XXdXd': 244, "xxx'xxXxx": 553, 'XXPSx': 332,
                      'XxxXxxxx.xxx': 181, 'XxxxXPSx': 577, "'xx": 17, '..': 533, 'xx.x': 593, 'x!dd': 518,
                      'XxXPSx': 602, 'XXxX': 243, 'xxxx="xxxx': 4, 'xxx;xxxx;Xxxxx': 546, 'xxx-d-xxxx-d-xx': 275,
                      'xxxx!x': 568, 'X&xxx;xxx;X.': 463, 'Xxxxx_Xxxxx': 524, 'XxxxXSSxxx': 494, 'xxx-dddd': 329,
                      'xXPxxxx': 287, 'dd/ddd': 359, 'XxxxxXxx': 127, 'xxxx.xxx.xx': 363, 'Xx.X.': 47, "xxxx'Xxxxx": 613,
                      'Xxxxx-dddx': 600, 'xdd': 445, 'ddd,ddd-xxxx': 157, 'Xxxxx-': 337, 'dd,ddd-x': 495, 'xx.': 268,
                      "xxxx,'Xxx": 486, "X'Xxxxx": 128, ':': 27, 'XdXd': 158, 'xxxx@xxxx.xx.xx': 435, 'XxxXxxxx': 68,
                      'xxxxXXX': 579, 'xxxx,"Xxxxx': 366, 'XxxX.': 82, 'XXX-d': 450, 'xxxx="dd': 75, 'dd-xxx': 79,
                      'x,x,PSxdx,x,C/Xxx': 509, 'XxXxxxXxxxx.xxx': 249, 'XXXxXxxXxxxx': 536, 'xxx?x': 566,
                      'xxxx-ddxx': 469, 'XdXX': 235, 'xxxx-': 53, '!': 16, 'xxxx?"Xxx': 487, 'd,d': 432, '(': 122,
                      'XXxxx': 341, 'X 1/4 xxx': 538, 'xxXd': 455, 'xxxxX': 272, 'xxX"': 472, 'xxxxXxxx': 621,
                      "d'd": 313, 'XPSdd': 339, 'd-xx-d': 482, 'XXdx': 383, 'XxxXdxxx': 617, 'XxXx': 252, 'XxX.': 534,
                      'dd.d.dd': 258, '-d.d': 204, 'xxxx,"Xxx': 356, 'd.d': 35, '.xxxx': 605, 'xXxxx': 136,
                      'X.X.-xxxx': 124, '+dd': 365, 'XXX*X.': 439, 'xXxx': 69, 'XxxXSSx': 475, 'XxX': 114, 'ddxxx': 314,
                      'Xxxxx?XX': 516, 'Xx.': 60, 'xxX.': 543, 'dx': 221, 'XXXxxxx': 147, 'x.x.xxxxx': 563, 'ddX.': 261,
                      '   ': 405, 'dd.d': 31, 'dd,ddd-xxx': 218, 'x+-': 441, 'Xxxxxxx': 270, 'XxxxxXxxxx': 156,
                      'XXd-xxxx': 402, 'xxxx="dddd': 406, 'Xxxxx+': 427, 'XxxxxXxxxXxxxx.xxx': 219, 'XxxxxX': 89,
                      'Xd': 134, 'XxxxXSSxxxx': 527, 'Xxx][xxxx': 503, "X'Xxxx": 96, 'dXxxxx': 346, 'Xd-xxxx': 401,
                      'Xxxx+': 398, 'xxxx]-xxxx': 311, "--xxx'x": 80, 'XxxxX!x': 424, 'xxd': 417, 'xxx,&xxx;xxxx': 552,
                      'ddd-Xxxxx': 349, 'dd,ddd-xxxx': 109, "x'X": 580, "'": 28, 'ddddxddd-xxxx': 437, 'XxxxXxX': 608,
                      '--': 19, 'Xxxxxddd': 433, 'xxxx.xxx.xxx': 493, 'xxXX.xxx': 326, 'd': 45, 'xxxx.&xxx;xxxx': 550,
                      'Xd-Xd': 189, 'd/dddxx': 324, "d'dd": 301, 'XXXx': 58, 'dd': 21, 'XxxxXXX': 467, "x'": 198,
                      'xxX-xxxX': 150, 'x.x.x': 312, "Xxxxx'xxxx": 581, 'XxxX!xxxx': 484, 'XxxXd.': 92, 'X.X.X.': 176,
                      'XxXXXXxxxx': 506, 'xxxx?Xxxxx': 478, 'XXdxX': 409, 'XxxxX!xx': 425, 'xxx-ddxx': 138, 'dd/dd': 139,
                      '%': 173, 'Xxxxxxxx': 320, 'd.d-xxxx': 242, 'XxX+-xxxx': 148, 'X&xxx;xxx;X': 462, 'xXXX': 70,
                      'X+': 174, "xxxx-'ddx": 118, "x'Xx": 606, 'X.-xxxx': 539, 'XxxxxXxXxxxx.xxx': 376, 'Xxxxx!XX': 481,
                      'XdX.': 251, 'XX,xxx': 570, 'XX-d': 345, 'xxxxdd.xxx': 572, ',': 9, 'xxx-dd-xxxx': 319,
                      'dd-xxxx': 37, 'ddxdd': 100, '[': 84, 'xxx.xxxx.xxx': 350, 'xXx': 292, 'xx?"X': 496, 'Xxx': 11,
                      'XXXXdXxxxx': 477, 'd.dd-xx': 119, '.dddd': 468, '.ddd': 205, 'XXd': 34, '*': 269, 'xxxX': 262,
                      'Xxxd': 436, 'Xxx.': 36, 'd.ddd': 57, 'xx': 7, 'xxxx-dx': 616, "Xx'xx": 65, 'xxXXXXxx': 423,
                      'dd/d/ddd': 357, 'xxxx-d/dd': 257, 'XxxxX.': 130, 'X.X.-Xxxxx': 456, 'XxxxxXPSxx': 592, "xx'": 322,
                      'd-X': 191, 'dd.ddd': 352, 'XXXXx': 86, 'dddX': 459, 'dd,dddd': 413, 'dd/ddxxx': 264,
                      'XX<<xX': 598, 'xX': 126, 'ddd,ddd': 51, "Xxxxx'xx": 470, 'd.dddd': 186, 'xXxxXx': 351,
                      'X&xxx;Xx': 200, 'XxxxxXSSxx': 590, 'xxxx?XX': 480, 'XXx': 116, '-xxxx': 492, 'XxxxxX!x': 535,
                      'Xxxxx.xxx': 87, 'x<<Xxxx': 400, 'd-xxxx': 220, 'ddd-xxxx': 73, 'XXPxxxx': 330, 'xxXxxx': 499,
                      'XxxxxXxxxx.xxxx': 610, "X'xxx": 187, 'XxxxXXXX': 308, 'xxxx./': 309, '(c)': 90, 'XXXdddd': 195,
                      'XxX-xx': 347, '/': 105, 'XxX+-x': 342, 'd,ddd,ddd-xxxx': 166, 'Xxxxx.xxxx': 184, "XX'xxxx": 603,
                      'Xxxxx?""Xxxxx': 489, '<<': 167, 'dd:dd': 153, 'Xxxxx-d': 271, 'XxxxxXX': 434,
                      'xxxxX 1/4 xxx': 284, 'ddd-xx': 403, 'XxxX"xx': 512, "dd-'dd": 393, 'Xddd': 296, 'xxxX!x': 210,
                      'd,ddd,ddd': 214, '@XxxxxXxxxXxxxx': 575, 'ddXxxxx.xxx': 386, 'Xxxxxdd': 361, 'X&xxx;X': 132,
                      'ddddxxxx.xxx': 410, 'dxX': 300, 'xx^d': 340, 'XxxXxxx': 78, 'X&xxx;X.': 67, 'Xxxx!x': 331,
                      'X-': 609, 'Xxxx.': 327, "Xxx'x": 237, 'Xxxx': 20, 'Xdx': 224, 'XXX.xxx': 233, '+d.dddd': 465,
                      'XdXxx': 192, 'ddX': 460, 'xxx-d-xxxx-d-xxxx': 278, 'ddd-XXXX': 279, 'Xxxxx.&xxx;xxxx': 547,
                      'Xx': 18, '---': 240, 'X.X.xx': 354, 'dXd': 238, 'XxxxxXxxxdddd': 528, 'xXxxxx': 99, 'd.dd': 117,
                      '@Xxxx': 531, '+': 144, 'd/dd/dddd': 253, 'xxxx?x': 232, 'XxxxxXxxxXxx': 50, 'ddd-XXX': 384,
                      'Xxxxxd': 569, 'X.X.': 29, 'XxXxxxxXxxxx.xxx': 325, 'XxxxxX$?xxxx': 286, 'XxXxxx.xxx': 179, '"': 3,
                      "xxx'xx": 390, 'XxxxXxxxx': 143, 'XxxxXxxxxXxxx': 102, 'dd.dddd': 230, 'dd,ddd-xx': 442,
                      'xxx-xxxx-xxx-xxxx-xxx.xxx': 106, 'xxxx-ddd': 416, 'd:dd:dd': 414, 'Xxxxx.xx': 399,
                      'xxxx?""Xxxxx': 488, 'Xxxxxdddd': 260, 'XxxxxX 1/4 xxx': 282, 'xxx.xxx': 502, 'xxx-d': 449,
                      'xxXXX': 245, 'ddd.d': 395, 'Xddx': 367, '\n': 52, 'xX"xxxx': 304, 'XXXXxxx': 146,
                      'XxxxXC/xx': 110, 'XX"': 597, 'XXXX.xxx': 259, 'd-Xxxxx': 225, 'dd:d': 236, 'XxxxxX+-x': 612,
                      'XXX': 30, 'ddd.ddd': 353, 'XxxX!x': 348, 'dd-': 48, 'XxxxxXxxxx.xxx': 95, 'XXPxx': 283,
                      'XdXdd': 565, 'xxxx--': 254, 'XxxXx': 111, 'xXdxXxxxXxxX': 294, "X'xx": 362, 'd:dd': 141,
                      'x-': 288, 'XxxxXxx': 94, 'XXXxXxx': 588, 'x,x,PSxdx,x,C/.': 510, 'X"x': 66, 'XxxXxx': 164,
                      'X.X.X.X.': 474, 'XxxX': 83, '$': 41, 'dddd': 43, 'Xxxxx,&xxx;xxxx': 555, 'XxxxxXXX': 142,
                      'xx-': 594, 'XxxxX"xxx': 582, 'xxxxXdx': 573, 'X-ddX': 444, '-ddd': 247, '/x/.': 379, 'xxX': 514,
                      'd,ddd-xxx': 217, 'XXXd-xxxx': 446, 'XxxxxX.': 267, "X'xxxx": 241, 'Xxxxx-ddx': 601,
                      'X.X.X.X.X.': 419, '@Xxx': 532, 'XxXxxxx': 76, 'dddd-xxxx': 137,
                      'xxxx-xx-xxx-xx-xxx-xxxx-xxxx.xxx': 104, "Xxxxx'x": 239, 'dd-Xxxx': 426, 'XxXxxXxx': 604,
                      'XxXXXX': 372, 'xxxxXXXx': 578, '#': 180, 'xxxx="ddd': 177, 'xxx;xxxx;xxx': 554, 'dxx': 185,
                      'ddd,ddd,ddd': 457, 'xxxxxxx': 504, 'XxxxXxxxx.xxx': 250, 'XxXxXx': 371, 'dddx': 263, '>': 5,
                      'XxxxxddXXX': 548, "x'x": 26, 'xx-ddd': 385, 'dd.dd': 302, 'x--': 255, 'xxxxXxxxx': 507,
                      'XxxxxXxxx.xxx': 407, 'XC/XX': 448, 'x.xxxx': 315, 'XxxXXXXxxx': 508, 'XXxxxX!x': 458,
                      'XxxxX': 131, 'Xx-d': 561, '?': 23, '@': 511, "'xxxx": 120, 'xxxxXSSxx': 589, 'x<<Xxx': 404,
                      'ddxx-': 212, 'XXXX@xxxx': 226, '|': 273, "'Xxxxx": 168, '  ': 188, 'd-Xx': 248, 'xxx-ddddx': 159,
                      'Xxx-': 140, 'XxxxxXSSx': 591, 'Xxxxx-dx': 619, 'xxx/xx': 163, 'XxxXxXx': 338, 'XXdd': 54,
                      'xxXXXX': 451, 'XXxxxx': 418, 'xXXXx': 72, 'xxxx.xxxx.xxx': 559, 'XxxXd': 91, 'ddddx': 71,
                      'xxxxXXXX': 620, 'XXX-': 560, 'XXXxXXX': 382, 'XXxxXXX': 368, 'd/dd,dddxx': 135, "Xx'Xxxxx": 540,
                      'dX.': 162, '...': 39, "xxxx'": 103, 'xxx-ddx': 149, 'X.X': 483, 'Xdd': 125, 'XXXXxxx.xxx': 364,
                      'dd-xx-dd-xxxx': 213, 'XXXXxX': 369, "Xxxxx'X": 202, 'XxxxX"xx': 491, '^': 307, '--X': 277,
                      'XxXX': 201, 'XxxxXSSxx': 614, 'dd.ddd.ddd.ddd': 452, '+d': 466, 'X.': 56, '--xxx': 172,
                      'XxxxXxxx': 152, '-dd': 246, 'XXXXd': 387, 'XxX,xxxx': 585, 'XxxxxXdx': 576, 'dxdd': 123,
                      'XxxXXX': 529, 'd.dd-xxxx': 318, 'XxX 1/4 -xxx': 520, 'XxXC/xxxx': 290, 'XXX-dd': 397,
                      '    ': 476, 'xxxx,"Xxx\'x': 274, 'XxxXdx': 93, 'XxxXxxxxXxxx': 447, 'xxxx': 8,
                      'xxx;xxxx;Xxx.&xxx;xxxx': 542, 'XXXxxx': 207, 'xxxxdxxxx': 396, 'XXXXxx': 358, '.': 10,
                      "xxx-'ddx": 113, 'xxx;xxxx;Xxxx': 551, 'xxxx/': 305, 'xxxxx': 196, "Xxx'xx": 321, 'xxX-xx': 558,
                      'XXXxxxx/"Xxxxx': 564, 'Xxxx-d/dd': 228, 'x.x.': 97, 'xxX"x': 490, 'XXxx': 206, 'dXX': 88,
                      'XXdxxxx': 607, 'Xxxxx@xxxx': 215, "XXX'X": 171, 'XXXdd': 154, 'XxX-': 599, 'XxxxxX.xxx': 557,
                      "XX'XX": 522, 'XxxxxX,x': 584, 'd/d': 389, '&': 74, '""Xx': 485, 'd,ddd-xxxx': 61,
                      'xxxxXXX.xxx': 537, 'xx?Xxxxx': 500, 'xxxxxxxx': 293, 'ddd.dd': 182, 'X.X.-Xxxx': 530,
                      'XXXXxxxx': 151, 'xxX"xx': 303, "x'Xxxxx": 295, 'XX.xxx': 336, 'XXXXxXX': 373, 'XxX<<x': 443,
                      "XXXX'x": 377, 'xxxx.x.xx': 431, 'd,dddxx': 231, 'xxxx.xxx/x/xxx.xxx': 428, 'X.X./Xx': 505,
                      'X-dXX': 190, 'XXxxxXd': 596, "dddd,'dd": 360, 'xX.': 615, 'XXX?XX': 517, 'dddxx': 199,
                      'Xxxxxx': 211, "'x": 12, 'XxxxxX,': 587, 'xxx?XX': 515, '@xxxx': 310, "xxxx'xx": 208,
                      'xX 1/4 x': 281, 'XXXxXx': 374, 'X-d': 234, 'XSSx': 344, 'X': 15, 'X-ddd': 391, "Xx'xxx": 388,
                      'd/dxxx': 265, 'xxxxXxx': 471, 'X!Xx': 583, "x'xxxx": 108, 'dd,ddd': 49, 'XXXxXxxxx': 394,
                      'XXXd': 355, 'XxXPxxx': 375, 'xxx/': 306, 'dX': 145, 'XPSd': 498, 'd,ddd': 44, ')': 115,
                      'dxxxx': 378, 'xxxx-d-xx': 276, 'xXXXX': 381, 'd/dxx': 323, 'XxxXxxxxXxxx.xxx': 408, ']': 85,
                      'XxX,xx': 611, 'XxxxxXx': 77, 'XXXXddd': 595, 'xxx-': 121, '=': 194, 'X-dd': 24,
                      'xxx;xxxx;Xx': 544, 'XX-dd': 160, 'XxxxXPxxxx': 266, 'xxxx="d': 2, 'XxxxXX': 101, "xxxx'x": 412,
                      '.d': 193, 'xxxx.xxx': 33, 'd/dd': 98, 'xxx;xxxx;XXX': 549, 'ddxxxXx': 229, 'dddd/dddd': 438,
                      "dddd-'dd": 392, "xxx'xxx": 415, 'dd/d': 59, ';': 14, 'XxxxxXxxxxXxxXxxxx.xxx': 421,
                      'XxxxX!xxxx': 501, 'XxX!xxx': 161, 'xxxxX$?xxxx': 285, 'xx-dd': 175, 'XXXXxxxxXxx': 525,
                      'Xxxxx': 6, 'X.X.x': 299, 'xxxx-d': 165, 'xxx.&xxx;xxxx': 545, "XxxxX'xxx": 454, 'XX-ddd': 513,
                      'ddXXX': 430, 'Xxxx.xxx': 209, 'XxX+-xx': 335, 'x.': 63, 'XxxXxxxxXxxxx.xxx': 420, 'xxx--': 291,
                      'xxxXxxxx': 289, 'XxxXXXX': 556, 'XXXXxX.': 370, '/x/': 380, 'XxXxxx': 62, 'XxXxxxx.xxx': 112,
                      'XX-ddx': 216, 'XX!xxxx': 298, 'Xxxx"X"Xx': 256, "xxx'x": 411, 'XxxxxXPxxxx': 280,
                      'XdddXXddXXX': 429, 'XXXxx': 334},
            'tag': {'WDT': 35, '-LRB-': 45, 'RBR': 33, 'WP$': 44, 'PDT': 40, 'IN': 17, 'SYM': 49, 'VBN': 19, ':': 18,
                    'JJR': 32, 'PRP$': 22, 'CC': 10, 'POS': 30, '_SP': 41, 'PRP': 6, 'EX': 29, '.': 9, 'AFX': 43,
                    'WP': 20, 'VBD': 21, 'LS': 47, 'NFP': 39, 'VBP': 5, 'FW': 37, 'ADD': 48, 'WRB': 34, "''": 4,
                    'NNS': 3, 'TO': 15, 'JJS': 36, 'VBG': 27, 'RB': 7, 'NN': 14, 'VB': 16, 'CD': 23, 'NNP': 1,
                    'XX': 0, ',': 8, '``': 2, 'DT': 12, 'RBS': 38, 'JJ': 13, '-RRB-': 46, 'UH': 26, '$': 42, 'RP': 24,
                    'NNPS': 31, 'MD': 25, 'VBZ': 11, 'HYPH': 28},
            'pos': {'VERB': 4, 'CCONJ': 7, 'ADJ': 10, 'SCONJ': 13, 'NUM': 14, 'SYM': 17, 'X': 0, 'DET': 9, 'ADP': 12,
                    'AUX': 8, 'ADV': 6, 'NOUN': 3, 'PROPN': 1, 'INTJ': 15, 'PUNCT': 2, 'PRON': 5, 'SPACE': 16, 'PART': 11},
            'ent_type': {'PRODUCT': 2, 'NONE': 0, 'LANGUAGE': 18, 'GPE': 7, 'CARDINAL': 6, 'ORDINAL': 11, 'TIME': 4,
                         'ORG': 1, 'PERCENT': 13, 'NORP': 12, 'LAW': 17, 'MONEY': 15, 'WORK_OF_ART': 14, 'QUANTITY': 8,
                         'EVENT': 16, 'DATE': 5, 'LOC': 10, 'PERSON': 3, 'FAC': 9},
            'ent_iob': {'I': 2, 'B': 1, 'O': 0},
            'sense': {'38': 19, '19': 15, '11': 10, '37': 48, '39': 43, '34': 16, '28': 26, '59': 54, '30': 32, '06': 9,
                      '50': 52, '23': 21, '02': 2, '24': 31, '17': 27, '16': 37, '44': 44, '14': 5, '52': 49, '05': 3,
                      '15': 14, '13': 18, '29': 39, '08': 8, '36': 45, '40': 41, '41': 42, '03': 4, '46': 17, '31': 46,
                      '12': 12, '35': 22, '33': 33, '01': 1, '18': 28, '25': 34, '45': 50, '42': 24, '55': 47, '10': 13,
                      '22': 29, '07': 7, '04': 6, '51': 53, '21': 30, '43': 20, '26': 23, '20': 25, '47': 51, '27': 35,
                      '32': 36, '09': 11, '48': 38, 'none': 0, '49': 40},
            'sentiment': {'positive': 1, 'none': 0, 'negative': 2}}

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

