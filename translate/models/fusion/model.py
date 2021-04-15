from torchtext import data

from configuration import cfg, device
from models.transformer.model import Transformer


class DictionaryFusionTransformer(Transformer):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        super(DictionaryFusionTransformer, self).__init__(SRC, TGT)
        print("TODO: [DictionaryFusionTransformer] modify the necessary class functions here!")

    def forward(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False, **kwargs):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        :return decoding_initializer_result_tensor (output of softmax),
                decoding_initializer_max_attention_indices (can be None),
                decoding_initializer_cumulative_loss,
                decoding_initializer_loss_size,
                batch_tokens_count
        """
        if not test_mode:
            bilingual_dict = kwargs['bilingual_dict']
            # TODO use the bilingual dict in here
            # You may want to convert 'bilingual_dict' into a torch tensor
        return self.decode(input_tensor_with_lengths, output_tensor_with_length, test_mode, beam_size=self.beam_size, **kwargs)
