
class DecodingSearchNode:
    def __init__(self, _id, decoder_context, next_token, attention_context, eos_predicted, coverage_vectors,
                 max_attention_indices, cumulative_loss, loss_size, cumulative_predicted_targets,
                 predicted_target_lm_score):
        """
        :param decoder_context: the encoder hidden vectors which can be used by the decoder in attention mechanism if
            any available
        :param attention_context: the computed attention vector based on the initial value of the :return next_token:\
            which has already been filled with start of the sentence token
        :param eos_predicted: a boolean vector tracking whether the end of sentence has already been predicted in in
            the sentence. This vector is useful in keeping track of the decoding and halting the decoding loop once all
            the sentences in the batch have generated the eos token.
        :param coverage_vectors: a matrix place holder for collecting the attentions scores over the input sequences in
            the batch
        :param predicted_target_lm_score: a place holder to collect the language model scores predicted for the current
            step predictions in the decoding process
        :param cumulative_predicted_targets: a place holder to collect the target sequence predictions step by step as
            the decoding proceeds
        :param max_attention_indices: a matrix place holder for collecting the source token ids gaining maximum values
            of attention in each decoding step
        :param cumulative_loss: a place holder for the cumulative loss of the decoding as the iterations proceed
        :param loss_size: the counter of decoding steps the loss of which is collected in :return cumulative_loss:
        """
        self.decoder_context = decoder_context
        self.next_token = next_token
        self.attention_context = attention_context
        self.eos_predicted = eos_predicted
        self.coverage_vectors = coverage_vectors
        self.cumulative_predicted_targets = cumulative_predicted_targets
        self.max_attention_indices = max_attention_indices
        self.cumulative_loss = cumulative_loss
        self.loss_size = loss_size
        self.predicted_target_lm_score = predicted_target_lm_score
        self.result_output = None
        self.result_query = None
        self.result_context = None
        self._id = _id

    @property
    def id(self):
        return self._id

    def set_result(self, output, query, decoder_context):
        self.result_output = output
        self.result_query = query
        self.result_context = decoder_context
