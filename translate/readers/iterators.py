"""
This file contains the customized torchtext data iterators
"""
import torchtext
if torchtext.__version__.startswith('0.9'):
    from torchtext.legacy import data
else:
    from torchtext import data
from configuration import cfg


class MyIterator(data.Iterator):
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
                created_batch = data.Batch(minibatch, self.dataset, self.device)
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
                if cfg.augment_input_with_bilingual_dict:
                    created_batch.data_args["bilingual_dict"] = [
                        [[cfg.lex(cfg.bilingual_dictionary, src_token, trg_token) for trg_token in mb.trg]
                         for src_token in mb.src] for mb in minibatch]
                yield created_batch
            if not self.repeat:
                return

    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class MyBucketIterator(data.BucketIterator):
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
                created_batch = data.Batch(minibatch, self.dataset, self.device)
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
