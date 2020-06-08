"""
This is a sample script on how pretrained BertWordPieceTokenizer vocab files are created, you may modify the hard-coded script and change any part
 depending on the desired output, you may also alter the predefined config flags, although it is NOT recommended to set [strip_accents=True]
"""
from tokenizers import BertWordPieceTokenizer
from sacremoses import MosesPunctNormalizer, MosesTokenizer


class MosesPreTokenizer:
    def __init__(self, lng, do_lowercase):
        self.mpn = MosesPunctNormalizer()
        self.moses_tokenizer = MosesTokenizer(lang=lng)
        self.do_lowercase = do_lowercase

    def pre_tokenize(self, text):
        return self.moses_tokenizer.tokenize(self.mpn.normalize(text.lower() if self.do_lowercase else text))


if __name__ == '__main__':
    lang = 'fr'
    clean_text = True
    handle_chinese_chars = True
    strip_accents = False
    lowercase = True
    vocab_size = 30000
    min_frequency = 2
    spt = ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "[UNK]", "[SEP]", "[CLS]", "[PAD]", "[MASK]"]
    if lang == "fr":
        train_data = "../.data/wmt19_de_fr/train.fr"
    elif lang == "en":
        train_data = "../.data/wmt19_en_de/train.en"
    else:
        raise ValueError("Undefined language {}".format(lang))

    tokenizer = BertWordPieceTokenizer(clean_text=clean_text, lowercase=lowercase,
                                       handle_chinese_chars=handle_chinese_chars, strip_accents=strip_accents)
    tokenizer.pre_tokenizer = MosesPreTokenizer(lang, lowercase)

    # Customize training
    print("Starting to train ...")
    tokenizer.train(files=train_data, vocab_size=vocab_size, show_progress=True, min_frequency=min_frequency, special_tokens=spt)
    # Save files to disk
    tokenizer.save(".", "moses-pre-tokenized-wmt-uncased-{}".format(lang))
