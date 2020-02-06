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
        tag = token.tag_ if len(token.tag_) else "none"
        shape = token.shape_
        ent_type = token.ent_type_ if len(token.ent_type_) else "none"
        ent_iob = token.ent_iob_ if len(token.ent_type_) else "o"
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


def project_sub_layers_trainer(file_adr, bert_tokenizer):
    """
    Implementation of the sub-layer model trainer which pre-trains the transformer heads using the BERT vectors.
    """
    # bert_lm = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    # model = torch.nn.Sequential(nn.Linear(D_in, H), nn.Linear(H, D_out)).to(device)
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
    print(vocabs)


if __name__ == '__main__':
    nlp = spacy.load("en")
    bert_tknizer = BertTokenizer.from_pretrained(model_name)
    if not int(sys.argv[1]):
        projection_trainer(sys.argv[2], bert_tknizer)
    else:
        project_sub_layers_trainer(sys.argv[2], bert_tknizer)

