"""
This file contains the customized torchtext data iterators
"""
import random
from configuration import cfg
from .data.itr import Iterator
from .data.batch import Batch
from .data.utils import batch

def pool(data, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
         random_shuffler=None, shuffle=False, sort_within_batch=False):
    """Sort within buckets, then batch, then shuffle batches.

    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle
    for p in batch(data, batch_size * 100, batch_size_fn):
        p_batch = batch(sorted(p, key=key), batch_size, batch_size_fn) \
            if sort_within_batch \
            else batch(p, batch_size, batch_size_fn)
        if shuffle:
            for b in random_shuffler(list(p_batch)):
                yield b
        else:
            for b in list(p_batch):
                yield b

class MyIterator(Iterator):
    """
    The customized torchtext iterator suggested in https://nlp.seas.harvard.edu/2018/04/03/attention.html
    The iterator is meant to speed up the training by token-wise batching
    """
    def __len__(self):
        return 0.0

    def __iter__(self):
        """This code is taken from torchtext.data.Iterator"""
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                created_batch = Batch(minibatch, self.dataset, self.device)
                created_batch.data_args = {}
                if cfg.augment_input_with_bert_src_vectors:  # this flag is an internal flag and is not set through configurations
                    # This is solely for efficiency purposes, although its not a good idea to combine model logic with input reader!
                    max_len = max(created_batch.src[1]).item()
                    bert_input_sentences = [self.dataset.src_tokenizer.tokenizer.convert_tokens_to_ids(mb.src) +
                                            [self.dataset.src_tokenizer.tokenizer.pad_token_id] * (max_len - len(mb.src)) for mb in minibatch]
                    created_batch.data_args["bert_src"] = bert_input_sentences
                if cfg.augment_input_with_syntax_infusion_vectors:
                    max_len = max(created_batch.src[1]).item()
                    syntax_data = [self.dataset.src_tokenizer.syntax_infused_container.convert(
                        self.dataset.src_tokenizer.detokenize(mb.src), max_len) for mb in minibatch]
                    for tag in self.dataset.src_tokenizer.syntax_infused_container.features_list:
                        created_batch.data_args["si_"+tag] = [s[tag] for s in syntax_data]
                yield created_batch
            if not self.repeat:
                return

    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in batch(d, self.batch_size * 100):
                    p_batch = batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class MyBucketIterator(Iterator):

    def create_batches(self):
        if self.sort:
            self.batches = batch(self.data(), self.batch_size,
                                 self.batch_size_fn)
        else:
            self.batches = pool(self.data(), self.batch_size,
                                self.sort_key, self.batch_size_fn,
                                random_shuffler=self.random_shuffler,
                                shuffle=self.shuffle,
                                sort_within_batch=self.sort_within_batch)

    def __iter__(self):
        """This code is taken from torchtext.data.Iterator"""
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                created_batch = Batch(minibatch, self.dataset, self.device)
                created_batch.data_args = {}
                if cfg.augment_input_with_bert_src_vectors:  # this flag is an internal flag and is not set through configurations
                    # This is solely for efficiency purposes, although its not a good idea to combine model logic with input reader!
                    max_len = max(created_batch.src[1]).item()
                    bert_input_sentences = [self.dataset.src_tokenizer.tokenizer.convert_tokens_to_ids(mb.src) +
                                            [self.dataset.src_tokenizer.tokenizer.pad_token_id] * (max_len - len(mb.src)) for mb in minibatch]
                    created_batch.data_args["bert_src"] = bert_input_sentences
                if cfg.augment_input_with_syntax_infusion_vectors:
                    max_len = max(created_batch.src[1]).item()
                    syntax_data = [self.dataset.src_tokenizer.syntax_infused_container.convert(
                        self.dataset.src_tokenizer.detokenize(mb.src), max_len) for mb in minibatch]
                    for tag in self.dataset.src_tokenizer.syntax_infused_container.features_list:
                        created_batch.data_args["si_"+tag] = [s[tag] for s in syntax_data]
                yield created_batch
            if not self.repeat:
                return
