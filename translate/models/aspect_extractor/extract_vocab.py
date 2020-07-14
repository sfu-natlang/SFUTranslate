"""
The code in this script is intended for running linguistic aspect extraction using pre-trained linguistic models.
"""
import unidecode
import torch

from tqdm import tqdm
from collections import Counter

from textblob import TextBlob
from nltk.wsd import lesk

from readers.tokenizers import SpacyTokenizer
from readers.sequence_alignment import extract_monotonic_sequence_to_sequence_alignment
from models.aspect_extractor.ae_utils import create_empty_linguistic_vocab
from configuration import device


def extract_linguistic_aspect_values(line, bert_tokenizer, spacy_tokenizer_1, spacy_tokenizer_2,
                                     required_features_list=("c_pos", "f_pos", "subword_shape", "ent_type", "ent_iob", "sense", "subword_position", "dependency_tag")):
    result = []
    lesk_queries = {"NOUN": 'n', "VERB": 'v', "ADJ": 'a', "ADV": 'r'}
    doc = spacy_tokenizer_1.tokenizer(line)
    spacy_doc = [token.text for token in doc]
    bert_doc = [token for token in bert_tokenizer.tokenize(line)]
    sp_bert_doc = spacy_tokenizer_2.tokenizer(" ".join(bert_doc))
    # spacy_labeled_bert_tokens = [token.text for token in sp_bert_doc]
    # assert len(spacy_labeled_bert_tokens) == len(bert_doc), "{}\n{}".format(spacy_labeled_bert_tokens, bert_doc)
    fertilities = extract_monotonic_sequence_to_sequence_alignment(spacy_doc, bert_doc)
    bert_doc_pointer = 0
    for token, fertility in zip(doc, fertilities):
        pos = token.pos_
        tag = token.tag_ if len(token.tag_) else "NONE"
        dep = token.dep_ if len(token.dep_) else "NONE"
        shape = token.shape_
        ent_type = token.ent_type_ if len(token.ent_type_) else "NONE"
        ent_iob = token.ent_iob_ if len(token.ent_type_) else "O"
        if "sense" in required_features_list:
            sense_data = lesk(spacy_doc, token.text, lesk_queries[pos] if pos in lesk_queries else "")
            sense = sense_data.name().split(".")[-1] if sense_data is not None else "none"
        else:
            sense = None
        if "sentiment" in required_features_list:
            sentiment = 'positive' if TextBlob(token.text).sentiment.polarity > 0.05 else 'negative' \
                if TextBlob(token.text).sentiment.polarity < -0.05 else 'none'
        else:
            sentiment = "none"
        linguistic_features = {"c_pos": pos, "f_pos": tag, "subword_shape": unidecode.unidecode(shape), "ent_type": ent_type,
                               "ent_iob": ent_iob, "sense": sense, "sentiment": sentiment, "dependency_tag": dep}
        for f_index in range(fertility):
            if fertility == 1:
                bis = "single"
            elif f_index == 0:
                bis = "begin"
            else:
                bis = "inside"
            lfc = linguistic_features.copy()
            if "token" in required_features_list:
                lfc["token"] = bert_doc[bert_doc_pointer]
            lfc["subword_shape"] = unidecode.unidecode(sp_bert_doc[bert_doc_pointer].shape_)
            lfc["subword_position"] = bis
            result.append(lfc)
            bert_doc_pointer += 1
        del linguistic_features
    return result


def extract_linguistic_vocabs(dataset_instance, bert_tokenizer, lang, lowercase_data):
    spacy_tokenizer_1, spacy_tokenizer_2 = SpacyTokenizer(lang, lowercase_data), SpacyTokenizer(lang, lowercase_data)
    spacy_tokenizer_2.overwrite_tokenizer_with_split_tokenizer()
    vocabs = create_empty_linguistic_vocab()
    vocab_cnts = {"c_pos": Counter(), "f_pos": Counter(), "subword_shape": Counter(), "ent_type": Counter(),
                  "ent_iob": Counter(), "sense": Counter(), "sentiment": Counter(), "subword_position": Counter(), "dependency_tag": Counter()}
    vocab_totals = {"c_pos": 0., "f_pos": 0., "subword_shape": 0., "ent_type": 0., "ent_iob": 0., "sense": 0., "sentiment": 0.,
                    "subword_position": 0., "dependency_tag": 0.}
    for input_sentence in tqdm(dataset_instance):
        sent = " ".join(input_sentence.src)
        res = extract_linguistic_aspect_values(sent, bert_tokenizer, spacy_tokenizer_1, spacy_tokenizer_2)
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


def dataset_iterator(dataset_object, b_size):
    """
    :param dataset_object: an instance of classes extending readers.datasets.generic.TranslationDataset
    :param b_size: the batch size of the returning sentence batches
    """
    res = []
    for tokenized_src_tgt in dataset_object:
        src_tokens = tokenized_src_tgt.src
        res.append(" ".join(src_tokens))
        if len(res) == b_size:
            yield res
            del res[:]
    yield res


def extract_aspects_and_weights(sent, linguistic_vocab, bert_tokenizer, spacy_tokenizer_1, spacy_tokenizer_2, required_features_list, padding_value=0):
    res = extract_linguistic_aspect_values(sent, bert_tokenizer, spacy_tokenizer_1, spacy_tokenizer_2, required_features_list)
    # the first index is <PAD>
    sent_extracted_features = [[linguistic_vocab[elem][token_feature[elem]][0] + 1 for token_feature in res] for elem in required_features_list]
    sent_extracted_feature_weights = [[linguistic_vocab[elem][token_feature[elem]][1] for token_feature in res] for elem in required_features_list]
    for ind in range(len(required_features_list)):
        sent_extracted_features[ind].insert(0, padding_value)
        sent_extracted_feature_weights[ind].insert(0, 0.)
        sent_extracted_features[ind].append(padding_value)
        sent_extracted_feature_weights[ind].append(0.)  # make sure you don't sum over this one
    return [torch.tensor(sent_extracted_features[ind], device=device).long() for ind in range(len(required_features_list))], \
           [torch.tensor(sent_extracted_feature_weights[ind], device=device).float() for ind in range(len(required_features_list))]


def map_sentences_to_vocab_ids(input_sentences, required_features_list, linguistic_vocab, spacy_tokenizer_1, spacy_tokenizer_2, bert_tokenizer,
                               padding_value=0):
    # will be a list of #input_sentences size each item a pair of size two and each item of the pair of size len(required_features_list)
    # len(input_sentences) * 2 * len(required_features_list)
    extracted_features_and_weights = [extract_aspects_and_weights(sent, linguistic_vocab, bert_tokenizer, spacy_tokenizer_1, spacy_tokenizer_2,
                                                                  required_features_list, padding_value) for sent in input_sentences]
    return [torch.nn.utils.rnn.pad_sequence([extracted_features_and_weights[sent_id][0][ind]for sent_id in range(len(input_sentences))],
                                            batch_first=True, padding_value=padding_value) for ind in range(len(required_features_list))], \
           [torch.nn.utils.rnn.pad_sequence([extracted_features_and_weights[sent_id][1][ind]for sent_id in range(len(input_sentences))],
                                            batch_first=True, padding_value=0.) for ind in range(len(required_features_list))]
