"""
The standard Transformer Encoder-Decoder architecture exactly as explained in the "attention is all you need paper".
"""
from translate.backend.utils import backend

__author__ = "Hassan S. Shavarani"


class EncoderDecoder(backend.nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        The module will run the input first through the :param encoder: module which returns a normalized tensor, 
         the encoded tensor then goes through the :param decoder: module which return an embedded target tensor which 
          then needs to go through the :param generator: to get mapped back to the vocabulary space of the target side. 
           :param src_embed: and :param tgt_embed: are the embedding layer instances which will be used to convert the 
            input and target tensors (containing sparse one-hot ids) into dense input and target embedding spaces. 
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
