"""
This is the trainer function which reads the training data and trains the aspect extractor module.
"""
import torch
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from readers.tokenizers import SpacyTokenizer
from configuration import device
from models.aspects.module import AspectExtractor

from models.aspects.extract_vocab import map_sentences_to_vocab_ids
from models.aspects.ae_utils import merge_subword_labels, create_reverse_linguistic_vocab, print_classification_report
from utils.init_nn import weight_init

try:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    from transformers import BertForMaskedLM
except ImportError:
    warnings.warn("transformers package is not available, transformers.BertForMaskedLM will not be accessible.")
    BertForMaskedLM = None


def create_train_report_and_persist_modules(model, save_model_name, all_actual_sw, all_prediction_sw, feature_pred_correct_all,
                                            feature_pred_corrects, required_features_list, resolution_strategy):
    print("reporting sub-word-level classification accuracy scores")
    print_classification_report(required_features_list, all_actual_sw, all_prediction_sw)
    for ind, feat in enumerate(required_features_list):
        print("{} sub-word level classification precision [collected]: {:.2f}%".format(
            feat.upper(), float(feature_pred_corrects[ind] * 100) / feature_pred_correct_all))
    print("reporting word-level classification accuracy scores")
    all_actual, all_prediction = merge_subword_labels(all_actual_sw, all_prediction_sw, required_features_list,
                                                      resolution_strategy=resolution_strategy)
    print_classification_report(required_features_list, all_actual, all_prediction)

    torch.save({'model': model}, save_model_name+".extractor")
    torch.save({'features_list': required_features_list, 'softmax': nn.Softmax(dim=-1), 'aspect_vectors': model.encoders,
                'bert_weights': model.bert_weights_for_average_pooling}, save_model_name)


def aspect_extractor_trainer(data_itr, model_name, bert_tokenizer, linguistic_vocab, required_features_list, lang, lowercase_data, H, lr,
                             scheduler_patience_steps, scheduler_decay_factor, scheduler_min_lr, epochs, max_norm, no_improvement_tolerance=5000,
                             save_model_name="project_sublayers.pt", relative_sizing=False, resolution_strategy="first", report_every=5000):
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
    assert len(Hs) > 0
    Hs.append(max(Hs))
    weight_ratio = int(float(H)/sum(Hs))
    assert weight_ratio > 1
    Hs = [int(weight_ratio * ind) for ind in Hs]
    Hs[-1] += max(0, (H - sum(Hs)))
    print("Loading the pre-trained BertForMaskedLM model: {}".format(model_name))
    bert_lm = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    number_of_bert_layers = len(bert_lm.bert.encoder.layer) + 1
    D_in = D_out = bert_lm.bert.pooler.dense.in_features
    reverse_linguistic_vocab = create_reverse_linguistic_vocab(linguistic_vocab)
    print("Loading Spacy Tokenizers")
    spacy_tokenizer_1, spacy_tokenizer_2 = SpacyTokenizer(lang, lowercase_data), SpacyTokenizer(lang, lowercase_data)
    spacy_tokenizer_2.overwrite_tokenizer_with_split_tokenizer()
    print("Creating the model")
    model = AspectExtractor(D_in, Hs, D_out, [len(linguistic_vocab[f]) + 1 for f in required_features_list],
                       number_of_bert_layers, required_features_list, reverse_linguistic_vocab).to(device)
    model.apply(weight_init)
    opt = optim.SGD(model.parameters(), lr=float(lr), momentum=0.9)
    scheduler = ReduceLROnPlateau(opt, mode='min', patience=scheduler_patience_steps, factor=scheduler_decay_factor,
                                  threshold=0.001, verbose=False, min_lr=scheduler_min_lr)
    print("Starting to train ...")
    break_condition = False
    for t in range(epochs):
        if break_condition:
            print("Minimum {} batches have been observed without any accuracy improvements in classifiers, ending the training ...".format(
                no_improvement_tolerance))
            break
        all_loss = 0.0
        all_tokens_count = 0.0
        feature_pred_corrects = [0 for _ in range(len(required_features_list))]
        feature_pred_correct_all = 0.0
        all_prediction = [[] for _ in required_features_list]
        all_actual = [[] for _ in required_features_list]
        # TODO use the actual dataset object instead of this iterator
        itr = data_itr()
        tolerance_counts = [0 for _ in required_features_list]
        tolerance_bests = [0.0 for _ in required_features_list]
        for batch_id, input_sentences in enumerate(itr):
            sequences = [torch.tensor(bert_tokenizer.tokenizer.encode(input_sentence, add_special_tokens=True), device=device)
                         for input_sentence in input_sentences]
            features, feature_weights = map_sentences_to_vocab_ids(
                input_sentences, required_features_list, linguistic_vocab,  spacy_tokenizer_1, spacy_tokenizer_2, bert_tokenizer)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=bert_tokenizer.tokenizer.pad_token_id)
            outputs = bert_lm(input_ids, masked_lm_labels=input_ids)[2]  # (batch_size * [input_length + 2] * 768)
            all_layers_embedded = torch.cat([o.detach().unsqueeze(0) for o in outputs], dim=0)
            embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0),
                                    model.softmax(model.bert_weights_for_average_pooling))
            # sequence_length, batch_size, len(feats)
            predictions = torch.zeros(embedded.size(1), embedded.size(0), len(required_features_list))
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
                _, loss, feature_pred_correct, feat_predictions = model(x, features_selected, feature_weights_selected)
                predictions[s] = feat_predictions
                for ind, score in enumerate(feature_pred_correct):
                    feature_pred_corrects[ind] += score.sum().item()
                feature_pred_correct_all += feature_pred_correct[0].size(0)
                model.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                opt.step()
                all_loss += loss.item()
                all_tokens_count += x.size(0)
                _classification_report_ = ["{}:{:.2f}%".format(feat.upper(), float(feature_pred_corrects[ind] * 100)
                                                               / feature_pred_correct_all) for ind, feat in enumerate(required_features_list)]
                itr.set_description("Epoch: {}, Average Loss: {:.2f}, [{}]".format(t, all_loss / all_tokens_count,
                                                                                   "; ".join(_classification_report_)))
            # if model has not had any improvements in any of the classifier scores after {no_improvement_tolerance} batches, the training will stop.
            for ind, feat in enumerate(required_features_list):
                feat_score = round(float(feature_pred_corrects[ind] * 100) / feature_pred_correct_all, 3)
                if tolerance_bests[ind] < feat_score:
                    tolerance_bests[ind] = feat_score
                    tolerance_counts[ind] = 0
                else:
                    tolerance_counts[ind] = tolerance_counts[ind] + 1
            break_condition = sum([1 if tolerance_counts[ind] >= no_improvement_tolerance else 0 for ind, feat in enumerate(
                required_features_list)]) == len(required_features_list)
            if break_condition:
                break
            scheduler.step(all_loss / all_tokens_count)
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
                        actual_id = int(features[idx][b][l].item()) - 1
                        predicted_label = reverse_linguistic_vocab[required_features_list[idx]][pred_id] if pred_id > -1 else '__PAD__'
                        actual_label = reverse_linguistic_vocab[required_features_list[idx]][actual_id] if actual_id > -1 else '__PAD__'
                        # predicted_bis, predicted_label = separate_bis_label(predicted_label)
                        # actual_bis, actual_label = separate_bis_label(actual_label)
                        if actual_label != '__PAD__':
                            all_actual[idx].append(actual_label)
                            all_prediction[idx].append(predicted_label)
                        # print(pred_tag, actual_label, actual_bis, predicted_label, predicted_bis, predicted_label == actual_label)
            if batch_id and batch_id % report_every == 0:
                print("Creating report/persisting trained model ...")
                create_train_report_and_persist_modules(model, save_model_name, all_actual, all_prediction, feature_pred_correct_all,
                                                        feature_pred_corrects, required_features_list, resolution_strategy)
        create_train_report_and_persist_modules(model, save_model_name, all_actual, all_prediction, feature_pred_correct_all,
                                                feature_pred_corrects, required_features_list, resolution_strategy)
    print("Training done.")



