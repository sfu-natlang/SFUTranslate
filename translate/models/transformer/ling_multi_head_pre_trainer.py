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
batch_size = 64
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
        print("Error:\n{}\n{}\n{}\n{}".format(spacy_doc, bert_doc, sum(fertilities), len(bert_doc)))
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
                features_selected = [f.select(1, s) for f in features]
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
            'pos': {'INTJ': 15, 'PROPN': 1, 'PUNCT': 2, 'NOUN': 3, 'VERB': 4, 'PRON': 5, 'ADV': 6, 'CCONJ': 7, 'AUX': 8,
                    'X': 0, 'DET': 9, 'ADJ': 10, 'PART': 11, 'ADP': 12, 'SCONJ': 13, 'NUM': 14, 'SPACE': 16, 'SYM': 17},
            'tag': {'XX': 0, 'NNP': 1, '``': 2, 'NNS': 3, "''": 4, 'VBP': 5, 'PRP': 6, 'RB': 7, ',': 8, '.': 9,
                    'CC': 10, 'VBZ': 11, 'DT': 12, 'JJ': 13, 'NN': 14, 'TO': 15, 'VB': 16, 'IN': 17, ':': 18,
                    'VBN': 19, 'WP': 20, 'VBD': 21, 'PRP$': 22, 'CD': 23, 'RP': 24, 'MD': 25, 'UH': 26, 'VBG': 27,
                    'HYPH': 28, 'EX': 29, 'POS': 30, 'NNPS': 31, 'JJR': 32, 'RBR': 33, 'WRB': 34, 'WDT': 35, 'JJS': 36,
                    'FW': 37, 'RBS': 38, 'NFP': 39, 'PDT': 40, '_SP': 41, '$': 42, 'AFX': 43, 'WP$': 44, '-LRB-': 45,
                    '-RRB-': 46, 'LS': 47, 'ADD': 48, 'SYM': 49},
            'shape': {'<': 0, 'xxx': 1, 'xxxx="d': 2, '"': 3, 'xxxx="xxxx': 4, '>': 5, 'Xxxxx': 6, 'xx': 7, 'xxxx': 8,
                      ',': 9, '.': 10, 'Xxx': 11, "'x": 12, 'x': 13, ';': 14, 'X': 15, '!': 16, "'xx": 17, 'Xx': 18,
                      '--': 19, 'Xxxx': 20, 'dd': 21, '-': 22, '?': 23, 'X-dd': 24, 'ddd': 25, "x'x": 26, ':': 27,
                      "'": 28, 'X.X.': 29, 'XXX': 30, 'dd.d': 31, 'XX': 32, 'xxxx.xxx': 33, 'XXd': 34, 'd.d': 35,
                      'Xxx.': 36, 'dd-xxxx': 37, '/xxx': 38, '...': 39, ' ': 40, '$': 41, 'XXXX': 42, 'dddd': 43,
                      'd,ddd': 44, 'd': 45, 'ddxx': 46, 'Xx.X.': 47, 'dd-': 48, 'dd,ddd': 49, 'XxxxxXxxxXxx': 50,
                      'ddd,ddd': 51, '\n': 52, 'xxxx-': 53, 'XXdd': 54, 'ddx': 55, 'X.': 56, 'd.ddd': 57, 'XXXx': 58,
                      'dd/d': 59, 'Xx.': 60, 'd,ddd-xxxx': 61, 'XxXxxx': 62, 'x.': 63, "xx'xx": 64, "Xx'xx": 65,
                      'X"x': 66, 'X&xxx;X.': 67, 'XxxXxxxx': 68, 'xXxx': 69, 'xXXX': 70, 'ddddx': 71, 'xXXXx': 72,
                      'ddd-xxxx': 73, '&': 74, 'xxxx="dd': 75, 'XxXxxxx': 76, 'XxxxxXx': 77, 'XxxXxxx': 78,
                      'dd-xxx': 79, "--xxx'x": 80, '.dd': 81, 'XxxX.': 82, 'XxxX': 83, '[': 84, ']': 85, 'XXXXx': 86,
                      'Xxxxx.xxx': 87, 'dXX': 88, 'XxxxXxx': 89, 'XxxxxXxxxx.xxx': 90, "X'Xxxx": 91, 'x.x.': 92,
                      'd/dd': 93, 'xXxxxx': 94, 'ddxdd': 95, 'XxxxXX': 96, 'XxxxXxxxxXxxx': 97, "xxxx'": 98,
                      'xxxx-xx-xxx-xx-xxx-xxxx-xxxx.xxx': 99, '/': 100, 'xxx-xxxx-xxx-xxxx-xxx.xxx': 101,
                      'xxx-xx-xx-xxx-xx-xxx-xx-xx-xxx.xxx': 102, "x'xxxx": 103, 'dd,ddd-xxxx': 104, 'XxxXx': 105,
                      'XxXxxxx.xxx': 106, "xxx-'ddx": 107, 'XxX': 108, ')': 109, 'XXx': 110, 'd.dd': 111,
                      "xxxx-'ddx": 112, 'd.dd-xx': 113, "'xxxx": 114, 'xxx-': 115, '(': 116, 'dxdd': 117,
                      'X.X.-xxxx': 118, 'Xdd': 119, 'xX': 120, 'XxxxxXxx': 121, "X'Xxxxx": 122, 'xxx-dd': 123,
                      'XxxxX.': 124, 'XxxxX': 125, 'X&xxx;X': 126, 'XxXxx': 127, 'Xd': 128, 'd/dd,dddxx': 129,
                      'xXxxx': 130, 'dddd-xxxx': 131, 'xxx-ddxx': 132, 'dd/dd': 133, 'Xxx-': 134, 'd:dd': 135,
                      'XxxxxXXX': 136, 'XxxxXxxxx': 137, '+': 138, 'dX': 139, 'XXXXxxx': 140, 'XXXxxxx': 141,
                      'xxx-ddx': 142, 'XXXXxxxx': 143, 'XxxxXxxx': 144, 'dd:dd': 145, 'XXXdd': 146, 'XxxxxXxxx': 147,
                      'XxxxxXxxxx': 148, 'ddd,ddd-xxxx': 149, 'XdXd': 150, 'xxx-ddddx': 151, 'XX-dd': 152, 'dX.': 153,
                      'xxx/xx': 154, 'XxxXxx': 155, 'xxxx-d': 156, 'd,ddd,ddd-xxxx': 157, '': 158, "'Xxxxx": 159,
                      'dd-xx': 160, "Xxx'xxxx": 161, "XXX'X": 162, '--xxx': 163, '%': 164, 'X+': 165, 'xx-dd': 166,
                      'X.X.X.': 167, '  ': 168, 'xxxx="ddd': 169, 'XxxXxxx.xxx': 170, 'XxXxxx.xxx': 171, '#': 172,
                      'XxxXxxxx.xxx': 173, 'ddd.dd': 174, 'X.x': 175, 'Xxxxx.xxxx': 176, 'dxx': 177, 'd.dddd': 178,
                      "X'xxx": 179, 'Xd-Xd': 180, 'X-dXX': 181, 'd-X': 182, 'XdXxx': 183, '.d': 184, '=': 185,
                      'XXXdddd': 186, "x'": 187, 'dddxx': 188, 'X&xxx;Xx': 189, 'XxXX': 190, "Xxxxx'X": 191,
                      '....': 192, '-d.d': 193, '.ddd': 194, 'XXxx': 195, 'XXXxxx': 196, "xxxx'xx": 197,
                      'Xxxx.xxx': 198, 'ddxx-': 199, 'dd-xx-dd-xxxx': 200, 'd,ddd,ddd': 201, 'Xxxxx@xxxx': 202,
                      'XX-ddx': 203, 'd,ddd-xxx': 204, 'dd,ddd-xxx': 205, 'XxxxxXxxxXxxxx.xxx': 206, 'd-xxxx': 207,
                      'dx': 208, 'XX&xxx;X': 209, '-d': 210, 'Xdx': 211, 'd-Xxxxx': 212, 'XXXX@xxxx': 213, 'XdX': 214,
                      'Xxxx-d/dd': 215, 'ddxxxXx': 216, 'dd.dddd': 217, 'd,dddxx': 218, 'XXX.xxx': 219, 'X-d': 220,
                      'XdXX': 221, 'dd:d': 222, "Xxx'x": 223, 'dXd': 224, "Xxxxx'x": 225, '---': 226, "X'xxxx": 227,
                      'd.d-xxxx': 228, 'XXxX': 229, 'XXdXd': 230, 'xxXXX': 231, '-dd': 232, '-ddd': 233, 'd-Xx': 234,
                      'XxXxxxXxxxx.xxx': 235, 'XxxxXxxxx.xxx': 236, 'XdX.': 237, 'XxXx': 238, 'd/dd/dddd': 239,
                      'xxxx--': 240, 'x--': 241, 'Xxxx"X"Xx': 242, 'xxxx-d/dd': 243, 'dd.d.dd': 244, 'XXXX.xxx': 245,
                      'Xxxxxdddd': 246, 'ddX.': 247, 'dddx': 248, 'dd/ddxxx': 249, 'd/dxxx': 250, 'XxxxxX.': 251,
                      'XxxxxX': 252, 'xx.': 253, '*': 254, 'Xxxxx-d': 255, 'xxxx,"Xxx\'x': 256, 'xxx-d-xxxx-d-xx': 257,
                      'xxxx-d-xx': 258, '--X': 259, 'xxx-d-xxxx-d-xxxx': 260, 'ddd-XXXX': 261, 'x-': 262,
                      'xxxXxxxx': 263, 'xxx--': 264, 'xXx': 265, 'Xddd': 266, 'xxxx-dd': 267, 'X.X.x': 268,
                      'dxX': 269, "d'dd": 270, 'dd.dd': 271, 'xxxx/': 272, 'xxx/': 273, '^': 274, 'XxxxXXXX': 275,
                      'xxxx./': 276, '@xxxx': 277, 'xxxx]-xxxx': 278, 'x.x.x': 279, "d'd": 280, 'ddxxx': 281,
                      'x.xxxx': 282, 'XxxxXxxxxXxx': 283, 'XXddd': 284, 'd.dd-xxxx': 285, 'xxx-dd-xxxx': 286,
                      "Xxx'xx": 287, "xx'": 288, 'd/dxx': 289, 'd/dddxx': 290, 'XxXxxxxXxxxx.xxx': 291, 'xxXX.xxx': 292,
                      'Xxxx.': 293, 'ddd-xxx': 294, 'xxx-dddd': 295, 'XXXxx': 296, 'XX.xxx': 297, 'Xxxxx-': 298,
                      'XxxXxXx': 299, 'PS': 300, 'xx^d': 301, 'XXxxx': 302, "x'xxx": 303, 'XX-d': 304, 'dXxxxx': 305,
                      'ddd-Xxxxx': 306, 'xxx.xxxx.xxx': 307, 'xXxxXx': 308, 'dd.ddd': 309, 'ddd.ddd': 310,
                      'X.X.xx': 311, 'XXXd': 312, 'xxxx,"Xxx': 313, 'dd/d/ddd': 314, 'XXXXxx': 315, 'dd/ddd': 316,
                      "dddd,'dd": 317, 'Xxxxxdd': 318, "X'xx": 319, 'xxxx.xxx.xx': 320, 'XXXXxxx.xxx': 321, '+dd': 322,
                      'xxxx,"Xxxxx': 323, 'Xddx': 324, 'XXxxXXX': 325, 'XXXXxX': 326, 'XXXXxX.': 327, 'XxXxXx': 328,
                      'XxXXXX': 329, 'XXXXxXX': 330, 'XXXxXx': 331, 'XxxxxXxXxxxx.xxx': 332, "XXXX'x": 333,
                      'dxxxx': 334, '/x/.': 335, '/x/': 336, 'xXXXX': 337, 'XXXxXXX': 338, 'XXdx': 339, 'ddd-XXX': 340,
                      'xx-ddd': 341, 'ddXxxxx.xxx': 342, 'XXXXd': 343, "Xx'xxx": 344, 'd/d': 345, "xxx'xx": 346,
                      'X-ddd': 347, "dddd-'dd": 348, "dd-'dd": 349, 'XXXxXxxxx': 350, 'ddd.d': 351, 'xxxxdxxxx': 352,
                      'XXX-dd': 353, 'Xxxx+': 354, 'Xxxxx.xx': 355, 'Xd-xxxx': 356, 'XXd-xxxx': 357, 'ddd-xx': 358,
                      '   ': 359, 'xxxx="dddd': 360, "x'Xxxxx": 361, 'XxxxxXxxx.xxx': 362, 'XxxXxxxxXxxx.xxx': 363,
                      'ddddxxxx.xxx': 364, "xxx'x": 365, "xxxx'x": 366, 'dd,dddd': 367, 'd:dd:dd': 368, "xxx'xxx": 369,
                      'xxxx-ddd': 370, 'xxd': 371, 'XXxxxx': 372, 'X.X.X.X.X.': 373, 'XxxXxxxxXxxxx.xxx': 374,
                      'XxxxxXxxxxXxxXxxxx.xxx': 375, 'Xdxdxdx': 376, 'xxXXXXxx': 377, 'dd-Xxxx': 378, 'Xxxxx+': 379,
                      'xxxx.xxx/x/xxx.xxx': 380, 'XdddXXddXXX': 381, 'ddXXX': 382, 'xxxx.x.xx': 383, 'd,d': 384,
                      'Xxxxxddd': 385, 'XxxxxXX': 386, 'xxxx@xxxx.xx.xx': 387, 'Xxxd': 388, 'ddddxddd-xxxx': 389,
                      'dddd/dddd': 390, 'XXX*X.': 391, 'dd,ddd-xx': 392, 'X-ddX': 393, 'xdd': 394, 'XXXd-xxxx': 395,
                      'XxxXxxxxXxxx': 396, 'xxx-d': 397, 'XXX-d': 398, 'xxXXXX': 399, 'dd.ddd.ddd.ddd': 400,
                      'X.X.-Xxxxx': 401, 'ddd,ddd,ddd': 402, 'dddX': 403, 'ddX': 404, 'xXX': 405, 'X&xxx;xxx;X': 406,
                      'X&xxx;xxx;X.': 407, 'Xxxx-': 408, '+d.dddd': 409, '+d': 410, 'XxxxXXX': 411, '.dddd': 412,
                      'xxxx-ddxx': 413, "Xxxxx'xx": 414, 'X.X.X.X.': 415, '    ': 416, 'XXXXdXxxxx': 417,
                      'xxxx?Xxxxx': 418, 'xxxx?XX': 419, 'Xxxxx!XX': 420, 'd-xx-d': 421, 'X.X': 422, '""Xx': 423,
                      "xxxx,'Xxx": 424, 'xxxx?"Xxx': 425, 'xxxx?""Xxxxx': 426, 'Xxxxx?""Xxxxx': 427, '-xxxx': 428,
                      'xxxx.xxx.xxx': 429, 'dd,ddd-x': 430, 'xx?"X': 431, '""Xxx': 432, 'xxXxxx': 433, 'xx?Xxxxx': 434,
                      'xxx.xxx': 435, 'Xxx][xxxx': 436, 'X.X./Xx': 437, 'XxXXXXxxxx': 438, 'xxxxXxxxx': 439,
                      'XxxXXXXxxx': 440, 'xxxxXxx': 441, 'xxxx.': 442, '@': 443, 'XX-ddd': 444, 'xxx?XX': 445,
                      'Xxxxx?XX': 446, 'XXX?XX': 447, 'EUR': 448, 'dd,dddxx': 449, 'dd,ddd,ddd': 450, "XX'XX": 451,
                      'XxxxxXxxXxxXXxxx': 452, 'Xxxxx_Xxxxx': 453, 'XXXXxxxxXxx': 454, 'XxxxxXxxxdddd': 455,
                      'XxxXXX': 456, 'X.X.-Xxxx': 457, '@Xxxx': 458, '@Xxx': 459, '..': 460, 'XxX.': 461,
                      'XXXxXxxXxxxx': 462, 'xxxxXXX.xxx': 463, 'X.-xxxx': 464, "Xx'Xxxxx": 465, 'xxX': 466,
                      'xxx;xxxx;Xxx.&xxx;xxxx': 467, 'xxX.': 468, 'xxx;xxxx;Xx': 469, 'xxx.&xxx;xxxx': 470,
                      'xxx;xxxx;Xxxxx': 471, 'Xxxxx.&xxx;xxxx': 472, 'XxxxxddXXX': 473, 'xxx;xxxx;XXX': 474,
                      'xxxx.&xxx;xxxx': 475, 'xxx;xxxx;Xxxx': 476, 'xxx,&xxx;xxxx': 477, 'xxx;xxxx;xxx': 478,
                      'Xxxxx,&xxx;xxxx': 479, 'XxxXXXX': 480, 'XxxxxX.xxx': 481, 'xxxx.xxxx.xxx': 482, 'XXX-': 483,
                      'Xx-d': 484, 'x.x.--xxxx': 485, 'XXXxxxx/"Xxxxx': 486, 'XdXdd': 487, 'Xxxxxd': 488, 'dddxxx': 489,
                      'xxxxdd.xxx': 490, '_': 491, '@XxxxxXxxxXxxxx': 492, 'xxxxXXXx': 493, 'xxxxXXX': 494,
                      "Xxxxx'xxxx": 495, 'XXXxXxx': 496, 'xx-': 497, 'XXXXddd': 498, 'Xxxxx-dddx': 499,
                      'Xxxxx-ddx': 500, 'XxXxxXxx': 501, '.xxxx': 502, "x'Xx": 503, 'X-': 504, 'XxxxxXxxxx.xxxx': 505,
                      "xxxx'Xxxxx": 506, 'xX.': 507, 'xxxx-dx': 508, "'X": 509, 'Xxxxx-dx': 510, 'xxxxXXXX': 511,
                      'xxxxXxxx': 512},
            'ent_type': {'NONE': 0, 'ORG': 1, 'PRODUCT': 2, 'PERSON': 3, 'TIME': 4, 'DATE': 5, 'CARDINAL': 6, 'GPE': 7,
                         'QUANTITY': 8, 'FAC': 9, 'LOC': 10, 'ORDINAL': 11, 'NORP': 12, 'PERCENT': 13,
                         'WORK_OF_ART': 14, 'MONEY': 15, 'EVENT': 16, 'LAW': 17, 'LANGUAGE': 18},
            'ent_iob': {'O': 0, 'B': 1, 'I': 2},
            'sense': {'none': 0, '01': 1, '02': 2, '05': 3, '03': 4, '14': 5, '04': 6, '07': 7, '08': 8, '06': 9,
                      '11': 10, '09': 11, '12': 12, '10': 13, '15': 14, '19': 15, '34': 16, '46': 17, '13': 18,
                      '38': 19, '43': 20, '23': 21, '35': 22, '26': 23, '42': 24, '20': 25, '28': 26, '17': 27,
                      '18': 28, '22': 29, '21': 30, '24': 31, '30': 32, '33': 33, '25': 34, '27': 35, '32': 36,
                      '16': 37, '48': 38, '29': 39, '49': 40, '40': 41, '41': 42, '39': 43, '44': 44, '36': 45,
                      '31': 46, '55': 47, '37': 48, '52': 49, '45': 50, '47': 51, '50': 52, '51': 53, '59': 54},
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

