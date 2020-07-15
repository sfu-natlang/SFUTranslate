"""
This script provides utility functions shared for different parts of aspect_extractor package.
"""
from sklearn.metrics import classification_report


class WordRepresentation:
    def __init__(self, actuals, preds, begin, end, required_features_list, tokens=None):
        self.rfl = required_features_list
        self.features_size = len(required_features_list)
        self._actuals = {}
        self._predictions = {}
        self._tokens = []
        for idx in range(len(required_features_list)):
            pred_tag = required_features_list[idx]
            self._actuals[pred_tag] = actuals[idx][begin:end]
            if pred_tag != "subword_shape" and pred_tag != "subword_position":
                assert all(x == self._actuals[pred_tag][0] for x in self._actuals[pred_tag])
            self._predictions[pred_tag] = preds[idx][begin:end]
        if tokens is not None:
            self._tokens = tokens[begin:end]

    def get_actual(self, tag):
        if tag != "subword_shape" or len(self._actuals[tag]) == 1:
            return self._actuals[tag][0]
        else:
            return self._actuals[tag][0] + "".join([x[2:] for x in self._actuals[tag][1:]])

    def get_pred(self, tag, resolution_strategy="first"):
        if tag != "subword_shape" or len(self._predictions[tag]) == 1:
            if resolution_strategy == "first":
                # heuristically returning the prediction of the first token for accuracy calculation
                return self._predictions[tag][0]
            elif resolution_strategy == "last":
                # heuristically returning the prediction of the last token for accuracy calculation
                return self._predictions[tag][-1]
            else:
                raise ValueError("Undefined resolution strategy: {}".format(resolution_strategy))
        else:
            return self._predictions[tag][0] + "".join([x[2:] for x in self._predictions[tag][1:]])


def extract_word_boundaries(bis_array):
    word_boundaries = []
    latest_multipart_word_start = -1
    latest_multipart_word_end = -1
    for idx, bis in enumerate(bis_array):
        if bis == "single":
            if latest_multipart_word_start != -1 and latest_multipart_word_end != -1:
                word_boundaries.append((latest_multipart_word_start, latest_multipart_word_end))
            elif latest_multipart_word_start != -1 and latest_multipart_word_end == -1:
                print("WARNING: latest_multipart_word_end was not found when the next single word began")
            # print("Single token word from [{}-{}]".format(idx, idx+1))
            word_boundaries.append((idx, idx+1))
            latest_multipart_word_start = -1
            latest_multipart_word_end = -1
        elif bis == "begin":
            if latest_multipart_word_start != -1 and latest_multipart_word_end != -1:
                word_boundaries.append((latest_multipart_word_start, latest_multipart_word_end))
            elif latest_multipart_word_start != -1 and latest_multipart_word_end == -1:
                print("WARNING: latest_multipart_word_end was not found when the next new word began")
            # print("Starting a new word from {}".format(idx))
            latest_multipart_word_start = idx
            latest_multipart_word_end = -1
        else:
            # print("Continuation of the word in {}".format(idx))
            latest_multipart_word_end = idx+1
    return word_boundaries


def merge_subword_labels(actuals, predictions, required_features_list, tokens=None, resolution_strategy="first"):
    assert "subword_position" in required_features_list
    bis_req_index = required_features_list.index("subword_position")
    word_boundaries = extract_word_boundaries(actuals[bis_req_index])
    words = [WordRepresentation(actuals, predictions, b, e, required_features_list, tokens) for b, e in word_boundaries]
    predictions = [[] for _ in required_features_list]
    actuals = [[] for _ in required_features_list]
    for idx, p_tag in enumerate(required_features_list):
        if idx == bis_req_index:
            actuals[idx] = ["single" for _ in words]
            predictions[idx] = ["single" for _ in words]
        else:
            actuals[idx] = [w.get_actual(p_tag) for w in words]
            predictions[idx] = [w.get_pred(p_tag, resolution_strategy) for w in words]
    return actuals, predictions


def create_empty_linguistic_vocab():
    return {"c_pos": {}, "f_pos": {}, "subword_shape": {}, "ent_type": {}, "ent_iob": {}, "sense": {}, "sentiment": {}, "subword_position": {}, "dependency_tag": {}}


def create_reverse_linguistic_vocab(ling_vocab):
    reverse_linguistic_vocab = create_empty_linguistic_vocab()
    for key in ling_vocab:
        for key2 in ling_vocab[key]:
            reverse_linguistic_vocab[key][ling_vocab[key][key2][0]] = key2
    return reverse_linguistic_vocab


def print_classification_report(required_features_list, all_actual, all_prediction):
    for idx in range(len(required_features_list)):
        pred_tag = required_features_list[idx]
        print('-' * 35 + pred_tag + '-' * 35)
        # target_names = list(linguistic_vocab[pred_tag].keys())
        report = classification_report(all_actual[idx], all_prediction[idx], output_dict=True,
                                       target_names=list(set(all_prediction[idx]+all_actual[idx])))['weighted avg']
        p = report['precision'] * 100
        r = report['recall'] * 100
        f = report['f1-score'] * 100
        s = report['support']
        print("Precision: {:.2f}%\tRecall: {:.2f}%\tF1: {:.2f}%\tSupport: {}".format(p, r, f, s))
        print('-' * 75)
