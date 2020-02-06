"""
The test script for training an intermediary sub-sectioned layer which does contain the same exact information as bert

"""
import sys
import spacy
from torch import optim
from torch import nn
import torch.nn.init as init
import torch
from transformers import BertTokenizer, BertForMaskedLM
from pathlib import Path
from tqdm import tqdm
from textblob import TextBlob
from nltk.wsd import lesk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ###############################################CONFIGURATIONS########################################################
model_name = 'bert-base-uncased'
desired_bert_layer = 12
D_in, H, D_out = 768, 1024, 768
epochs = 10
lr = 0.01
batch_size = 128
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


def spacy_to_bert_aligner(spacy_doc, bert_doc):
    """
    This function receives two lists extracted from spacy, and bert tokenizers on the same sentence,
    and returns the alignment fertilities from spacy to bert.
    the output will have a length equal to the size of spacy_doc each index of which indicates the number
    of times the spacy element characteristics must be copied to equal the length of the bert tokenized list.
    """
    len_spacy = len(spacy_doc)
    fertilities = [1] * len_spacy
    bert_f_pointer = 0
    spacy_legacy_token = ""
    bert_legacy_token = ""
    for s_i in range(len_spacy):
        spacy_token = spacy_legacy_token + spacy_doc[s_i]
        spacy_legacy_token = ""
        if spacy_token == bert_legacy_token:
            fertilities[s_i] = 0
            bert_legacy_token = ""
            continue
        if not len(spacy_token.strip()):
            fertilities[s_i] = 0
            continue
        if bert_f_pointer < len(bert_doc):
            bert_token = bert_doc[bert_f_pointer]
        else:
            raise ValueError()
        bert_token = bert_token[2:] if bert_token.startswith("##") else bert_token
        bert_token = bert_legacy_token + bert_token
        bert_legacy_token = ""
        if len(bert_token) > len(spacy_token):
            spacy_legacy_token = spacy_token
            bert_legacy_token = bert_token
            bert_f_pointer += 1
            continue
        while bert_token != spacy_token.lower():
            bert_f_pointer += 1
            if bert_f_pointer < len(bert_doc):
                tmp = bert_doc[bert_f_pointer]
            else:
                raise ValueError()
            bert_token += tmp[2:] if tmp.startswith("##") else tmp
            fertilities[s_i] += 1
        bert_f_pointer += 1
    assert sum(fertilities) == len(bert_doc), "Error:\n{}\n{}".format(spacy_doc, bert_doc)
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
        linguistic_features = {"pos": pos, "tag": tag, "shape": shape, "ent_type": ent_type,
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
            embedded = outputs[desired_bert_layer].detach()
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


def project_sub_layers_trainer(file_adr, bert_tokenizer, linguistic_vocab, required_features_list):
    """
    Implementation of the sub-layer model trainer which pre-trains the transformer heads using the BERT vectors.
    """
    # bert_lm = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    # model = torch.nn.Sequential(nn.Linear(D_in, H), nn.Linear(H, D_out)).to(device)

    pass


if __name__ == '__main__':
    nlp = spacy.load("en")
    bert_tknizer = BertTokenizer.from_pretrained(model_name)
    if not int(sys.argv[1]):
        projection_trainer(sys.argv[2], bert_tknizer)
    else:
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
        # TODO after being done with testing, you'd need to call extract_linguistic_vocabs() to extract the vocab
        required_features_list = ['pos', 'shape', 'tag']
        project_sub_layers_trainer(sys.argv[2], bert_tknizer, multi30k_linguistic_vocab, required_features_list)
