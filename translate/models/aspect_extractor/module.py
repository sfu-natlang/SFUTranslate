"""
This is the implementation of Aspect Extractor module with the aspect classification loss, reconstruction loss, and aspect vector similarity loss
 implemented in its training objective.
"""
import torch
from torch import nn

from random import random
from configuration import device


class AspectExtractor(torch.nn.Module):
    def __init__(self, D_in, Hs, D_out, feature_sizes, number_of_bert_layers, features_list, reverse_linguistic_vocab, padding_index=0):
        super(AspectExtractor, self).__init__()
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
        self.features_list = features_list
        self.reverse_linguistic_vocab = reverse_linguistic_vocab

    def verbose_results(self, ling_classes, features, feature_weights):
        if random() < self.verbose_debug_percentage:  # debug
            print("\n")
            for ind, lc in enumerate(ling_classes):
                class_type = self.features_list[ind]  # TODO remove this line it is not safe
                true_labels = features[ind]
                predicted_labels = lc.argmax(dim=-1)
                for e in range(true_labels.size(0)):
                    tl = true_labels[e].item()
                    if tl > 0:
                        true_label = self.reverse_linguistic_vocab[class_type][tl - 1]
                    else:
                        continue
                    pl = predicted_labels[e].item()
                    if pl > 0:
                        pred_label = self.reverse_linguistic_vocab[class_type][pl - 1]
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
        feat_predictions = torch.cat([lc.argmax(dim=-1).unsqueeze(0) for lc in ling_classes], dim=0).t()
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
        return y_pred, loss, feature_pred_correct, feat_predictions

    def sanity_test(self, x, class_id):
        encoded = [self.encoders[i](x) for i in range(len(self.encoders))]
        ling_classes = [self.feature_classifiers[class_id](encoded[i]) for i in range(len(self.encoders)-1) if i != class_id]
        feat_predictions = torch.cat([lc.argmax(dim=-1).unsqueeze(0) for lc in ling_classes], dim=0).t()
        return feat_predictions
