"""
The implementation of the transformer model in accordance to the Annotated Transformer
 (http://nlp.seas.harvard.edu/2018/04/03/attention.html) with an additional functionality to compute the loss value in 
  the forward path (the loss is added to module for speed optimization purposes). The module can be configured via 
   setting the following values in the config file to the desired values.
##################################################
trainer:
    model:
        type: transformer
        bsize: 64 # size of the training sentence batches
        init_val: 0.1 # the value to range of which random variables get initiated
        N: 6 # number of encoder/decoder layers
        d_model: 512 # size of each encoder/decoder layer
        d_ff: 2048 # size of intermediate layer in feedforward sub-layers
        h: 8 # number of heads
        dropout: 0.1 # the dropout used in encoder/decoder model parts
##################################################
"""
import copy
import numpy as np
from typing import List, Any, Tuple

from translate.configs.loader import ConfigLoader
from translate.learning.modelling import AbsCompleteModel
from translate.backend.utils import backend, Variable
from translate.learning.modules.mlp.feedforward import PositionwiseFeedForward
from translate.learning.modules.mlp.generator import GeneratorNN
from translate.learning.modules.transformer.attention import MultiHeadedAttention
from translate.learning.modules.transformer.criterion import LabelSmoothing
from translate.learning.modules.transformer.decoder import DecoderLayer, Decoder
from translate.learning.modules.transformer.encoder import EncoderLayer, Encoder
from translate.learning.modules.transformer.transformer import EncoderDecoder
from translate.learning.modules.transformer.utils import PositionalEncoding, Embeddings
from translate.readers.datareader import AbsDatasetReader

__author__ = "Hassan S. Shavarani"


class Transformer(AbsCompleteModel):
    def __init__(self, configs: ConfigLoader, train_dataset: AbsDatasetReader):
        """
        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param train_dataset: the dataset from which the statistics regarding dataset will be looked up during
         model configuration
        """
        super(Transformer, self).__init__(LabelSmoothing(len(train_dataset.target_vocabulary),
                                                         train_dataset.target_vocabulary.get_pad_word_index(),
                                                         smoothing=configs.get("trainer.model.smoothing", 0.1)))
        self.dataset = train_dataset
        self.batch_size = configs.get("trainer.model.bsize", must_exist=True)
        # init_val = configs.get("trainer.model.init_val", 0.01)
        N = configs.get("trainer.model.N", must_exist=True)
        d_model = configs.get("trainer.model.d_model", must_exist=True)
        d_ff = configs.get("trainer.model.d_ff", must_exist=True)
        h = configs.get("trainer.model.h", must_exist=True)
        dropout = configs.get("trainer.model.dropout", 0.1)

        self.max_length = train_dataset.max_sentence_length()
        self.sos_token_id = train_dataset.target_vocabulary.get_begin_word_index()
        self.eos_token_id = train_dataset.target_vocabulary.get_end_word_index()
        self.pad_token_id = train_dataset.target_vocabulary.get_pad_word_index()
        self.use_cuda = backend.cuda.is_available()

        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            backend.nn.Sequential(Embeddings(d_model, len(self.dataset.source_vocabulary)), c(position)),
            backend.nn.Sequential(Embeddings(d_model, len(self.dataset.target_vocabulary)), c(position)),
            GeneratorNN(d_model, len(self.dataset.target_vocabulary), dropout))
        if False:
            # Shared Embeddings: When using BPE with shared vocabulary we can share the same weight
            #  vectors between the source / target / generator.
            self.model.src_embed[0].lut.weight = self.model.tgt_embeddings[0].lut.weight
            self.model.generator.lut.weight = self.model.tgt_embed[0].lut.weight

        # This was important from their code to initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                backend.nn.init.xavier_uniform_(p)

    def forward(self, input_tensor: backend.Tensor, target_tensor: backend.Tensor, *args, **kwargs) \
            -> Tuple[backend.Tensor, int, List[Any]]:
        input_mask = args[-2]
        target_mask = args[-1]
        target_tensor_y = args[0]
        out = self.model(input_tensor, target_tensor, input_mask, target_mask)
        out = self.model.generator(out)
        loss = self.criterion(out.view(-1, out.size(-1)), target_tensor_y.view(-1))
        n_tokens = (target_tensor != self.pad_token_id).data.sum().item()
        # The decode function is not called while running the forward function due to optimization purposes.
        return loss, n_tokens, []

    def validate_instance(self, prediction_loss: float, hyp_ids_list: List[List[int]], input_id_list: backend.Tensor,
                          ref_ids_list: backend.Tensor, *args, **kwargs) -> Tuple[float, float, str]:
        """
        :param prediction_loss: the model calculated loss value over the current prediction
        :param hyp_ids_list: the current predicted Batch of sequences of ids. In case of this model the value is always
         an empty list (it has not been removed from the api to make the interface consistent with the other types of 
         model). This value can be safely ignored for this model since it will get computed inside this function
        :param input_id_list: the input batch over which the predictions are generated
        :param ref_ids_list: the expected Batch of sequences of ids
        :param args: contains the Transformer style mask tensors (as the last two indices of the args list)
        :return: the bleu score between the reference and prediction batches, in addition to a sample result
        """
        hyp_ids_list = []
        hyp_ids_tensor = self.greedy_decode(input_id_list, args[-2], self.max_length)[:, 1:]
        for sentence_index in range(hyp_ids_tensor.size(0)):
            sent = []
            for word in hyp_ids_tensor[sentence_index]:
                word = word.item()
                if word != self.pad_token_id:
                    sent.append(word)
                if word == self.eos_token_id:
                    break
            hyp_ids_list.append(sent)
        bleu_score, ref_sample, hyp_sample = self.dataset.compute_bleu(
            ref_ids_list[:, 1:], hyp_ids_list, ref_is_tensor=True,
            reader_level=self.dataset.get_target_word_granularity())
        result_sample = u"E=\"{}\", P=\"{}\"\n".format(ref_sample, hyp_sample)
        return bleu_score, prediction_loss, result_sample

    def optimizable_params_list(self) -> List[Any]:
        return [self.model.parameters()]

    def greedy_decode(self, src, src_mask, max_len) -> backend.Tensor:
        """
        The function in charge of performing the decoding for the transformer model (using the highest predicted 
         probability as the prediction in each step)
        :param src: the tensor containing the converted, padded batch of sentences to be translated
        :param src_mask: the transformer style mask generated for the source batch
        :param max_len: maximum allowed length of sentence to be generated 
        :return: the predicted target side ids for the sentences in the input batch   
        """
        memory = self.model.encode(src, src_mask)
        ys = backend.ones(src.size(0), 1).fill_(self.sos_token_id).type_as(src.data)
        for i in range(max_len - 1):
            out = self.model.decode(memory, src_mask, Variable(ys),
                                    Variable(self.subsequent_mask(ys.size(1)).type_as(src.data)))
            prob = self.model.generator(out[:, -1])
            _, next_word = backend.max(prob, dim=1)
            ys = backend.cat([ys, next_word.unsqueeze(1)], dim=1)
        return ys

    @staticmethod
    def subsequent_mask(size):
        """
        The function to create a square shaped mask tensor which masks the rest but first {i} words in the sequence for 
         the {i}th row of the mask tensor. an example output of this function follows for the :param size: of 3.  
         [[1 0 0]
          [1 1 0]
          [1 1 1]]
        """
        # TODO refactor this function into the TransformerReaderWrapper
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return backend.from_numpy(subsequent_mask) == 0
