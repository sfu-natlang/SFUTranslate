"""
This is the implementation of all necessary modules required for aspects package including:
 - Aspect Extractor with the aspect classification loss, reconstruction loss, and aspect vector similarity loss implemented in its training objective.
 - Aspect Integration module to be used in Aspect Augmented NMT model.
 - Multi-Head Aspect Augmentation Layer to be used in Multi-Head Aspect Augmented NMT model.
 - Multi-Headed Attention; a customized version of multi-head attention module which receives aspect vectors as well as token embeddings in input.
 - SyntaxInfusedSRCEmbedding; an embedding layer which combines the explicit syntactic information along with the token embeddings to create the final
      input embeddings for the syntax-infused NMT model.
 - BertEmbeddingIntegration; an embedding layer to replace the vanilla token embedding layer of transformer which feeds the source tokens into Bert
      and combines its values using a trainable weighted sum to create the input embedding for the NMT model. The bert embeddings are frozen and will
      not receive gradients during training.
"""
import torch
from torch import nn

from random import random
from configuration import cfg, device

from readers.data_provider import src_tokenizer_obj

from models.transformer.modules import SublayerConnection, PositionalEncoding, Embeddings
from models.transformer.utils import clones, attention

try:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    from transformers import BertForMaskedLM
except ImportError:
    warnings.warn("transformers package is not available, transformers.BertForMaskedLM will not be accessible.")
    BertForMaskedLM = None


class AspectExtractor(torch.nn.Module):
    def __init__(self, D_in, Hs, D_out, feature_sizes, number_of_bert_layers, features_list, reverse_linguistic_vocab, padding_index=0):
        super(AspectExtractor, self).__init__()
        self.equal_length_Hs = (sum([Hs[0] == h for h in Hs]) == len(Hs))
        self.consider_adversarial_loss = False
        self.encoders = nn.ModuleList([nn.Linear(D_in, h) for h in Hs])
        self.decoder = nn.Linear(sum(Hs), D_out)
        self.feature_classifiers = nn.ModuleList([nn.Linear(h, o) for h, o in zip(Hs[:-1], feature_sizes)])

        self.uniqueness_bridges = nn.ModuleList([nn.ModuleList([nn.Linear(other_aspect_h, class_feature_size)
                                                                if i != j else None for j, other_aspect_h in enumerate(Hs[:-1])])
                                                 for i, class_feature_size in enumerate(feature_sizes)])
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.class_loss_fn = nn.CrossEntropyLoss(ignore_index=padding_index, reduction='none')
        self.pair_distance = nn.PairwiseDistance(p=2)
        self.discriminator = nn.Linear(D_out, 1)

        self.bert_weights_for_average_pooling = nn.Parameter(torch.zeros(number_of_bert_layers), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        self.verbose_debug_percentage = 0.0000
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

    def uniqueness_forward(self, x, features, feature_weights):
        encoded = [self.encoders[i](x) for i in range(len(self.encoders))]
        loss = torch.zeros((1, 1), device=device)
        for class_id in range(len(self.encoders)-1):
            ling_classes = [self.uniqueness_bridges[class_id][i](encoded[i]) for i in range(len(self.encoders)-1) if i != class_id]
            for lc in ling_classes:
                mask = (feature_weights[class_id] != 0.).float()
                c_loss = self.class_loss_fn(lc, features[class_id])
                w = feature_weights[class_id]
                final_c_loss = (c_loss * mask) / (w+1e-32)
                loss += final_c_loss.sum()
        return loss

    def sanity_test(self, x, class_id):
        encoded = [self.encoders[i](x) for i in range(len(self.encoders))]
        ling_classes = [self.feature_classifiers[class_id](encoded[i]) for i in range(len(self.encoders)-1) if i != class_id]
        feat_predictions = torch.cat([lc.argmax(dim=-1).unsqueeze(0) for lc in ling_classes], dim=0).t()
        return feat_predictions

    def sanity_test2(self, x, class_id):
        encoded = [self.encoders[i](x) for i in range(len(self.encoders))]
        ling_classes = [self.uniqueness_bridges[class_id][i](encoded[i]) for i in range(len(self.encoders)-1) if i != class_id]
        feat_predictions = torch.cat([lc.argmax(dim=-1).unsqueeze(0) for lc in ling_classes], dim=0).t()
        return feat_predictions


class AspectIntegration(nn.Module):
    def __init__(self, d_model):
        super(AspectIntegration, self).__init__()
        self.d_model = d_model
        self.bert_lm = None
        self.aspect_vectors = None
        self.linguistic_embedding_to_d_model = None
        self.input_gate = nn.Linear(d_model * 2, d_model, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.bert_weights_for_average_pooling = None
        # self.number_of_bert_layers = 0

    def init_model_params(self):
        print("Running the init params for BERT language model")
        self.bert_lm = BertForMaskedLM.from_pretrained(src_tokenizer_obj.model_name, output_hidden_states=True).to(device)
        # self.number_of_bert_layers = len(self.bert_lm.bert.encoder.layer) + 1
        # self.bert_weights_for_average_pooling = nn.Parameter(torch.zeros(self.number_of_bert_layers), requires_grad=True)
        print("Running the init params for aspect_vectors")
        try:
            so = torch.load(cfg.aspect_vectors_data_address, map_location=lambda storage, loc: storage)
        except KeyError:
            print("AspectIntegration module failed to initialize since \"aspect_vectors_data_address\" is not set in the config file.\nExiting!")
            exit()
        self.aspect_vectors = so['aspect_vectors'].to(device)
        self.bert_weights_for_average_pooling = nn.Parameter(so['bert_weights'].to(device), requires_grad=True)
        aspect_vector_key_size = self.aspect_vectors[0].out_features
        aspect_vector_feature_count = len(self.aspect_vectors) - 1
        if aspect_vector_feature_count > 0:
            self.linguistic_embedding_to_d_model = nn.Linear(aspect_vector_key_size * aspect_vector_feature_count, self.d_model, bias=True).to(device)

    def forward(self, input_tensor, **kwargs):
        """
        :param input_tensor: the output of vanilla transformer embedding augmented with positional encoding information
        :param kwargs: must contain the bert indexed input token ids in a tensor named "bert_src"
        :return:
        """
        if "bert_src" in kwargs and kwargs["bert_src"] is not None:
            x_prime = self.linguistic_embedding_to_d_model(torch.cat(
                self.get_ling_embed_attention_keys_from_bert_converted_ids(kwargs["bert_src"]), dim=-1))
        else:
            raise ValueError("bert_src information are not provided!")
        return self.input_gate(torch.cat((x_prime, input_tensor), dim=-1))

    def get_ling_embed_attention_keys_from_bert_converted_ids(self, bert_input_sentences):
        input_ids = torch.tensor(bert_input_sentences, device=device)
        outputs = self.bert_lm(input_ids)[1]  # (batch_size * [input_length + 2] * 768)
        all_layers_embedded = torch.cat([o.unsqueeze(0) for o in outputs], dim=0)
        embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0),
                                self.softmax(self.bert_weights_for_average_pooling))  # [:, 1:-1, :]
        # ##############################################################################################################
        # len(features_list) * batch_size * max_sequence_length, (H/ (len(features_list) + 1))
        keys = [hc(embedded).detach() for hc in self.aspect_vectors[:-1]]  # the last layer contains what we have not considered
        return keys


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_aspects, d_left_over, dropout=0.1, value_from_token_embedding=False):
        """
        Implements Figure 2 (right) of the paper (https://arxiv.org/pdf/1706.03762.pdf)
        """
        super(MultiHeadedAttention, self).__init__()
        # assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.value_from_token_embedding = value_from_token_embedding
        print("Aspect multi-head attention \'V\' comes from {}".format("token embeddings" if value_from_token_embedding else "aspect embeddings"))
        self.query_linear = nn.Linear(d_model, self.d_k * self.h)
        self.aspect_linear = nn.Linear(d_aspects, self.d_k)
        if self.value_from_token_embedding:
            self.value_linear = nn.Linear(d_model, self.d_k * self.h)
        else:
            self.value_linear = nn.Linear(d_aspects, self.d_k)
        self.final_linear = nn.Linear(self.d_k * self.h, d_model)
        if d_left_over > 0:
            self.left_over_linear = nn.Linear(d_left_over, self.d_k)
        else:
            self.left_over_linear = None
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, aspect_vectors_list, mask=None, left_over=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        key = torch.cat([e.unsqueeze(1) for e in aspect_vectors_list], dim=1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        if not self.value_from_token_embedding:
            value = self.value_linear(key)
        else:
            value = self.value_linear(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        query = self.query_linear(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.aspect_linear(key)
        if self.left_over_linear is not None:
            assert left_over is not None
            lv = self.left_over_linear(left_over).unsqueeze(1)
            key = torch.cat((key, lv), dim=1)
            if not self.value_from_token_embedding:
                value = torch.cat((value, lv), dim=1)
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.final_linear(x)


class MultiHeadAspectAugmentationLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward
    """
    def __init__(self, use_left_over_vector=False, value_from_token_embedding=False):
        super(MultiHeadAspectAugmentationLayer, self).__init__()
        self.d_model = int(cfg.transformer_d_model)
        dropout = float(cfg.transformer_dropout)
        self.sublayer = clones(SublayerConnection(self.d_model, dropout), 2)
        self.aspect_attn = None
        self.bert_lm = None
        self.aspect_vectors = None
        self.bert_weights_for_average_pooling = None
        self.use_left_over_vector = use_left_over_vector
        print("Aspect multi-head attention will{} use the left-over aspect vector".format("" if use_left_over_vector else " not"))
        self.softmax = nn.Softmax(dim=-1)
        self.value_from_token_embedding = value_from_token_embedding

    def init_model_params(self):
        print("Running the init params for BERT language model")
        self.bert_lm = BertForMaskedLM.from_pretrained(src_tokenizer_obj.model_name, output_hidden_states=True).to(device)
        # self.number_of_bert_layers = len(self.bert_lm.bert.encoder.layer) + 1
        # self.bert_weights_for_average_pooling = nn.Parameter(torch.zeros(self.number_of_bert_layers), requires_grad=True)
        print("Running the init params for aspect_vectors")
        try:
            so = torch.load(cfg.aspect_vectors_data_address, map_location=lambda storage, loc: storage)
        except KeyError:
            print("MultiHeadAspectAugmentationLayer failed to initialize since \"aspect_vectors_data_address\" is not set in the config file.\n"
                  "Exiting!")
            exit()
        self.aspect_vectors = so['aspect_vectors'].to(device)
        self.bert_weights_for_average_pooling = nn.Parameter(so['bert_weights'].to(device), requires_grad=True)
        aspect_vector_key_size = self.aspect_vectors[0].out_features
        if self.use_left_over_vector:
            aspect_vector_feature_count = len(self.aspect_vectors)
            left_over_vector_key_size = self.aspect_vectors[-1].out_features
        else:
            aspect_vector_feature_count = len(self.aspect_vectors) - 1
            left_over_vector_key_size = 0
        self.aspect_attn = MultiHeadedAttention(aspect_vector_feature_count, self.d_model, aspect_vector_key_size, left_over_vector_key_size,
                                                value_from_token_embedding=self.value_from_token_embedding).to(device)

    def forward(self, x, mask, **kwargs):
        """
        Follow Figure 1 (left) for connections [https://arxiv.org/pdf/1706.03762.pdf]
        """
        if "bert_src" in kwargs and kwargs["bert_src"] is not None:
            x_prime, left_over = self.get_ling_embed_attention_keys_from_bert_converted_ids(kwargs["bert_src"])
        else:
            raise ValueError("bert_src information are not provided!")
        return self.sublayer[0](x, lambda x: self.aspect_attn(x, x_prime, mask, left_over))

    def get_ling_embed_attention_keys_from_bert_converted_ids(self, bert_input_sentences):
        input_ids = torch.tensor(bert_input_sentences, device=device)
        outputs = self.bert_lm(input_ids)[1]  # (batch_size * [input_length + 2] * 768)
        all_layers_embedded = torch.cat([o.unsqueeze(0) for o in outputs], dim=0)
        embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0),
                                self.softmax(self.bert_weights_for_average_pooling))  # [:, 1:-1, :]
        keys = [hc(embedded).detach() for hc in self.aspect_vectors[:-1]]  # the last layer contains what we have not considered
        return keys, self.aspect_vectors[-1](embedded).detach() if self.use_left_over_vector else None


class SyntaxInfusedSRCEmbedding(nn.Module):
    def __init__(self, src_vocab_len):
        super(SyntaxInfusedSRCEmbedding, self).__init__()
        from readers.data_provider import src_tokenizer_obj
        self.features_list = src_tokenizer_obj.syntax_infused_container.features_list
        assert len(self.features_list) > 0
        d_model = int(cfg.transformer_d_model)
        dropout = float(cfg.transformer_dropout)
        max_len = int(cfg.transformer_max_len)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len).to(device)
        self.token_embeddings = Embeddings(d_model, src_vocab_len).to(device)
        fd = src_tokenizer_obj.syntax_infused_container.features_dict
        # fd_sizes = {tag: len(fd[tag])for tag in self.features_list}
        self.syntax_embeddings = nn.ModuleList([Embeddings(int(d_model/len(self.features_list)), len(fd[tag])).to(device) for tag in self.features_list])
        self.syntax_embeddings_size = int(d_model/len(self.features_list)) * len(self.features_list)  # could be less than d_model
        self.infused_embedding_to_d_model = nn.Linear(self.syntax_embeddings_size + d_model, d_model, bias=True).to(device)

    def forward(self, input_tensor, **kwargs):
        """
        :param input_tensor: the output of vanilla transformer embedding augmented with positional encoding information
        :param kwargs: must contain the bert indexed input token ids in a tensor named "bert_src"
        :return:
        """
        syntactic_inputs = []
        for tag in self.features_list:
            lookup_tag = "si_"+tag
            if lookup_tag in kwargs and kwargs[lookup_tag] is not None:
                syntactic_inputs.append(torch.tensor(kwargs[lookup_tag], device=device).long())
            else:
                raise ValueError("The required feature tag {} information are not provided by the reader!".format(lookup_tag))
        syntactic_embedding = torch.cat([se(si) for se, si in zip(self.syntax_embeddings, syntactic_inputs)], dim=-1)
        token_embedding = self.token_embeddings(input_tensor)
        final_embedding = self.infused_embedding_to_d_model(torch.cat((syntactic_embedding, token_embedding), dim=-1))
        return self.positional_encoding(final_embedding)


class BertEmbeddingIntegration(nn.Module):
    def __init__(self, d_model):
        super(BertEmbeddingIntegration, self).__init__()
        self.d_model = d_model
        self.bert_lm = None
        self.softmax = nn.Softmax(dim=-1)
        self.bert_weights_for_average_pooling = None
        self.number_of_bert_layers = 0
        self.output_bridge = None

    def init_model_params(self):
        print("Running the init params for BERT language model")
        self.bert_lm = BertForMaskedLM.from_pretrained(src_tokenizer_obj.model_name, output_hidden_states=True).to(device)
        self.number_of_bert_layers = len(self.bert_lm.bert.encoder.layer) + 1
        self.bert_weights_for_average_pooling = nn.Parameter(torch.zeros(self.number_of_bert_layers), requires_grad=True).to(device)
        for p in self.bert_weights_for_average_pooling:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # TODO if final output size if not equal to self.d_model convert it back to self.d_model space using self.output_bridge
        lm_hidden_size = self.bert_lm.bert.config.hidden_size
        if self.d_model != lm_hidden_size:
            self.output_bridge = nn.Linear(lm_hidden_size, self.d_model).to(device)
            for p in self.output_bridge.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, **kwargs):
        """
        :param kwargs: must contain the bert indexed input token ids in a tensor named "bert_src"
        :return:
        """
        if "bert_src" in kwargs and kwargs["bert_src"] is not None:
            return self.get_ling_embed_attention_keys_from_bert_converted_ids(kwargs["bert_src"])
        else:
            raise ValueError("bert_src information are not provided!")

    def get_ling_embed_attention_keys_from_bert_converted_ids(self, bert_input_sentences):
        input_ids = torch.tensor(bert_input_sentences, device=device)
        outputs = self.bert_lm(input_ids)[1]  # (batch_size * [input_length + 2] * 768)
        all_layers_embedded = torch.cat([o.detach().unsqueeze(0) for o in outputs], dim=0)
        # TODO "self.bert_weights_for_average_pooling.to(device)" should not be done each time.
        embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0), self.softmax(self.bert_weights_for_average_pooling.to(device)))  # [:, 1:-1, :]
        if self.output_bridge is not None:
            embedded = self.output_bridge(embedded)
        return embedded