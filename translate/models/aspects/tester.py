"""
This is the test function which looks at the training data and reports the classification results of different trained aspects.
"""
import torch

from readers.tokenizers import SpacyTokenizer
from models.aspects.extract_vocab import map_sentences_to_vocab_ids
from models.aspects.ae_utils import merge_subword_labels, create_reverse_linguistic_vocab, print_classification_report
from configuration import device

try:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    from transformers import AutoModelForMaskedLM
except ImportError:
    warnings.warn("transformers package is not available, transformers.AutoModelForMaskedLM will not be accessible.")
    AutoModelForMaskedLM = None


def create_test_report(all_loss, all_tokens_count, all_actual_sw, all_prediction_sw, sanity_all_actual, sanity_all_prediction, feature_pred_correct_all,
                       feature_pred_corrects, required_features_list, check_result_sanity, resolution_strategy):
    if check_result_sanity:
        print("reporting sanity classification accuracy scores [the scores are expected be really low!]")
        print_classification_report(required_features_list, sanity_all_actual, sanity_all_prediction)

    print("reporting sub-word-level classification accuracy scores")
    print_classification_report(required_features_list, all_actual_sw, all_prediction_sw)
    for ind, feat in enumerate(required_features_list):
        print("{} sub-word level classification precision [collected]: {:.2f}%".format(
            feat.upper(), float(feature_pred_corrects[ind] * 100) / feature_pred_correct_all))
    print("reporting word-level classification accuracy scores")
    all_actual, all_prediction = merge_subword_labels(all_actual_sw, all_prediction_sw, required_features_list,
                                                      resolution_strategy=resolution_strategy)
    print_classification_report(required_features_list, all_actual, all_prediction)
    print(("Average Test Loss: {:.2f}".format(all_loss / all_tokens_count)))


def aspect_extractor_tester(data_itr, model_name, bert_tokenizer, linguistic_vocab, required_features_list, lang, lowercase_data,
                            load_model_name="generic_aspect_vectors.pt", resolution_strategy="first", check_result_sanity=False,
                            check_with_trained_sanity_heads=False):
    bert_lm = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    saved_obj = torch.load(load_model_name+".extractor", map_location=lambda storage, loc: storage)
    model = saved_obj['model'].to(device)
    reverse_linguistic_vocab = create_reverse_linguistic_vocab(linguistic_vocab)
    feature_pred_corrects = [0 for _ in range(len(required_features_list))]
    feature_pred_correct_all = 0.0
    all_prediction = [[] for _ in required_features_list]
    all_actual = [[] for _ in required_features_list]
    all_tokens = []
    if check_result_sanity:
        sanity_all_prediction = [[] for _ in required_features_list]
        sanity_all_actual = [[] for _ in required_features_list]
    else:
        sanity_all_actual = sanity_all_prediction = None
    # TODO use the actual dataset object instead of this iterator
    itr = data_itr()
    spacy_tokenizer_1, spacy_tokenizer_2 = SpacyTokenizer(lang, lowercase_data), SpacyTokenizer(lang, lowercase_data)
    spacy_tokenizer_2.overwrite_tokenizer_with_split_tokenizer()
    for input_sentences in itr:
        all_tokens.extend([tkn for input_sentence in input_sentences for tkn in bert_tokenizer.tokenize(input_sentence)])
        with torch.no_grad():
            all_loss = 0.0
            all_tokens_count = 0.0
            sequences = [torch.tensor(bert_tokenizer.tokenizer.encode(input_sentence), device=device)
                         for input_sentence in input_sentences]
            features, feature_weights = map_sentences_to_vocab_ids(
                input_sentences, required_features_list, linguistic_vocab, spacy_tokenizer_1, spacy_tokenizer_2, bert_tokenizer)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=bert_tokenizer.tokenizer.pad_token_id)
            if input_ids.size(1) > bert_lm.config.max_position_embeddings:
                continue
            outputs = bert_lm(input_ids, masked_lm_labels=input_ids)[2]  # (batch_size * [input_length + 2] * 768)
            all_layers_embedded = torch.cat([o.detach().unsqueeze(0) for o in outputs], dim=0)
            embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0),
                                    model.softmax(model.bert_weights_for_average_pooling))
            predictions = torch.zeros(embedded.size(1), embedded.size(0), len(required_features_list))
            if check_result_sanity:
                sanity_predictions = [torch.zeros(embedded.size(1), embedded.size(0), len(required_features_list) - 1
                                                  )] * len(required_features_list)
            for s in range(1, embedded.size(1)-1):
                x = embedded.select(1, s)
                # TODO make sure the f.select(1, -1) is a good idea also check why s < f.size(1) might happen (probably padding ?!?).
                features_selected = [f.select(1, s) if s < f.size(1) else f.select(1, -1) for f in features]
                feature_weights_selected = [fw.select(1, s) if s < fw.size(1) else fw.select(1, -1) for fw in feature_weights]
                _, loss, feature_pred_correct, feat_predictions = model(x, features_selected, feature_weights_selected)
                predictions[s] = feat_predictions
                if check_result_sanity:
                    for sanity_ind in range(len(required_features_list)):
                        if not check_with_trained_sanity_heads:
                            sanity_predictions[sanity_ind][s] = model.sanity_test(x, sanity_ind)
                        else:
                            sanity_predictions[sanity_ind][s] = model.sanity_test2(x, sanity_ind)
                for ind, score in enumerate(feature_pred_correct):
                    feature_pred_corrects[ind] += score.sum().item()
                feature_pred_correct_all += feature_pred_correct[0].size(0)
                all_loss += loss.item()
                all_tokens_count += x.size(0)

            predictions = predictions.transpose(0, 1)
            for b in range(predictions.size(0)):
                for l in range(1, predictions.size(1)-1):
                    classes = predictions[b][l]
                    for idx in range(len(required_features_list)):
                        pred_id = int(classes[idx].item()) - 1
                        if idx >= len(features) or b >= features[idx].size(0) or l >= features[idx].size(1):
                            # print("WARNING: skipping access to index out of bounds for a tensor with size "
                            #      "({}, {}, {}) with indices [{}, {}, {}]".format(len(features), features[idx].size(0),
                            #                                                      features[idx].size(1), idx, b, l))
                            continue
                        desired_linguistic_set = reverse_linguistic_vocab[required_features_list[idx]]
                        actual_id = int(features[idx][b][l].item()) - 1
                        predicted_label = desired_linguistic_set[pred_id] if pred_id > -1 else '__PAD__'
                        actual_label = desired_linguistic_set[actual_id] if actual_id > -1 else '__PAD__'
                        if actual_label != '__PAD__':
                            all_actual[idx].append(actual_label)
                            all_prediction[idx].append(predicted_label)
            if check_result_sanity:
                for sanity_ind in range(len(required_features_list)):
                    sanity_preds = sanity_predictions[sanity_ind].transpose(0, 1)
                    for b in range(sanity_preds.size(0)):
                        for l in range(1, sanity_preds.size(1)-1):
                            classes = sanity_preds[b][l]
                            idx = -1
                            for rf_idx in range(len(required_features_list)):
                                if rf_idx == sanity_ind:
                                    continue
                                else:
                                    idx += 1
                                pred_id = int(classes[idx].item()) - 1
                                if idx >= len(features) or b >= features[rf_idx].size(0) or l >= features[rf_idx].size(1):
                                    # print("WARNING: skipping access to index out of bounds for a tensor with size "
                                    #      "({}, {}, {}) with indices [{}, {}, {}]".format(len(features), features[idx].size(0),
                                    #                                                      features[idx].size(1), idx, b, l))
                                    continue
                                desired_linguistic_set = reverse_linguistic_vocab[required_features_list[rf_idx]]
                                if len(desired_linguistic_set) <= pred_id:  # should not happen!
                                    continue
                                actual_id = int(features[rf_idx][b][l].item()) - 1
                                predicted_label = desired_linguistic_set[pred_id] if pred_id > -1 else '__PAD__'
                                actual_label = desired_linguistic_set[actual_id] if actual_id > -1 else '__PAD__'
                                if actual_label != '__PAD__':
                                    sanity_all_actual[rf_idx].append(actual_label)
                                    sanity_all_prediction[rf_idx].append(predicted_label)
    # TODO if size of all_actual[idx] > 20M or size of all_prediction[idx] > 20M this will fail!
    create_test_report(all_loss, all_tokens_count, all_actual, all_prediction, sanity_all_actual, sanity_all_prediction, feature_pred_correct_all,
                       feature_pred_corrects, required_features_list, check_result_sanity, resolution_strategy)
