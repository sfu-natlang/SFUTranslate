from ..configs.loader import ConfigLoader


class Vocab:
    def __init__(self, configs: ConfigLoader):
        self.unk_word = configs.get("reader.vocab.unk_word", "<unk>")
        self.bos_word = configs.get("reader.vocab.bos_word", "<s>")
        self.eos_word = configs.get("reader.vocab.eos_word", "</s>")
        self.pad_word = configs.get("reader.vocab.pad_word", "<pad>")
        self.bpe_separator = configs.get("reader.vocab.bpe_separator", "@@")
