"""
Implementation of different tokenizers to be used by the data provider. The pre-trained tokenizers are intended to use
  the pre-trained vocabulary files distributed by huggingface.tokenizers which are trained over big corpora
    in each supported language.
"""
import spacy
from spacy.tokenizer import Tokenizer
from tokenizers import BertWordPieceTokenizer
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer
from requests import get
import os

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from transformers import BertTokenizer


class GenericTokenizer:
    """
    The very basic tokenizer mainly for debugging purposes
    """
    @staticmethod
    def tokenize(text):
        return text.split()

    @staticmethod
    def detokenize(tokenized_list):
        return " ".join(tokenized_list)

    @property
    def model_name(self):
        return "Generic"


class PreTrainedTokenizer(GenericTokenizer):
    vocab_files = {
        "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
        "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
        "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
        "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
        "bert-base-german-cased": "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
        "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
        "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
        "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
        "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
        "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt",
        "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-vocab.txt",
        "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-vocab.txt",
        "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/vocab.txt",
        "bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/vocab.txt",
        "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/vocab.txt",
    }

    def __init__(self, lang, root='.data', clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True):
        """
        Example instantiation: PreTrainedTokenizer("bert-base-uncased", root="../.data")
        """
        pre_trained_model_name = self.get_default_model_name(lang, lowercase)
        self._model_name_ = pre_trained_model_name
        if not os.path.exists(root):
            os.mkdir(root)
        assert pre_trained_model_name in self.vocab_files, \
            "The requested pre_trained tokenizer model {} does not exist!".format(pre_trained_model_name)
        url = self.vocab_files[pre_trained_model_name]
        f_name = root + "/" + os.path.basename(url)
        if not os.path.exists(f_name):
            with open(f_name, "wb") as file_:
                response = get(url)
                file_.write(response.content)
        self.moses_tkn = PyMosesTokenizer(lang, lowercase)
        self.tokenizer = BertWordPieceTokenizer(f_name, clean_text=clean_text, lowercase=lowercase,
                                                handle_chinese_chars=handle_chinese_chars, strip_accents=strip_accents)

    def tokenize(self, text):
        """
        You can recover the output of this function using " ".join(encoded_list).replace(" ##", "")
        :param text: one line of text in type of str
        :return a list of tokenized "str"s
        """
        if not len(text.strip()):
            return [""]
        # encoding = self.tokenizer.encode(n_text, add_special_tokens=False)
        encoding = self.tokenizer.encode_tokenized(self.moses_tkn.tokenize(text))
        # encoding contains "ids", "tokens", and "offsets"
        return encoding.tokens

    def detokenize(self, tokenized_list):
        # TODO make it work
        temp_result = []
        for token in tokenized_list:
            if len(temp_result) and token.startswith("##"):
                temp_result[-1] = temp_result[-1] + token[2:]
            else:
                temp_result.append(token)
        return self.moses_tkn.detokenize(temp_result)

    def decode(self, encoded_ids_list):
        """
        :param encoded_ids_list: list of int ids
        :return a decoded str
        """
        decoded = self.tokenizer.decode(encoded_ids_list)
        return decoded

    @staticmethod
    def get_default_model_name(lang, lowercase):
        if lang == "en" and lowercase:
            return "bert-base-uncased"
        elif lang == "en" and not lowercase:
            return "bert-base-cased"
        elif lang == "zh":
            return "bert-base-chinese"
        elif lang == "de" and lowercase:
            return "bert-base-german-dbmdz-uncased"
        elif lang == "de" and not lowercase:
            return "bert-base-german-dbmdz-cased"
        elif lang == "fi" and lowercase:
            return "bert-base-finnish-uncased-v1"
        elif lang == "fi" and not lowercase:
            return "bert-base-finnish-cased-v1"
        else:
            raise ValueError("No pre-trained tokenizer found for language {} in {} mode".format(
                lang, "lowercased" if lowercase else "cased"))

    @property
    def model_name(self):
        return self._model_name_


class PyMosesTokenizer(GenericTokenizer):
    """
    The call to standard moses tokenizer
    """
    def __init__(self, lang, lowercase):
        self.mpn = MosesPunctNormalizer()
        self.tokenizer = MosesTokenizer(lang=lang)
        self.detokenizer = MosesDetokenizer(lang=lang)
        self.lowercase = lowercase

    def tokenize(self, text):
        return self.tokenizer.tokenize(self.mpn.normalize(text.lower() if self.lowercase else text))

    def detokenize(self, tokenized_list):
        temp_result = ""
        t_list_len = len(tokenized_list)
        for t_ind, token in enumerate(tokenized_list):
            apos_cnd = token == "&apos;" and t_ind < t_list_len - 1 and tokenized_list[t_ind + 1] == "s"
            if apos_cnd or token == "/":
                temp_result = temp_result.strip() + token
            else:
                temp_result += token + " "
        return self.detokenizer.detokenize(temp_result.strip().split())

    @property
    def model_name(self):
        return "Moses"


class PTBertTokenizer:
    """
    The tokenizer pre-trained tokenizer trained alongside BERT by huggingface
    """
    def __init__(self, lang, lowercase=True):
        # the tokenizer names are the same for BertTokenizer and PreTrainedTokenizer since they have both been distributed by huggingface
        pre_trained_model_name = PreTrainedTokenizer.get_default_model_name(lang, lowercase)
        self.tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
        self.mpn = MosesPunctNormalizer()
        self.detokenizer = MosesDetokenizer(lang=lang)
        self._model_name_ = pre_trained_model_name

    def tokenize(self, text):
        return self.tokenizer.tokenize(self.mpn.normalize(text))

    def detokenize(self, tokenized_list):
        # WARNING! this is a one way tokenizer, the detokenized sentences do not necessarily align with the actual tokenized sentences!
        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokenized_list))

    @staticmethod
    def get_default_model_name(lang, lowercase):
        return PreTrainedTokenizer.get_default_model_name(lang, lowercase)

    @property
    def model_name(self):
        return self._model_name_


class SpacyTokenizer:
    """
    The very basic tokenizer mainly for debugging purposes
    """
    def __init__(self, lang):
        pre_trained_model_name = self.get_default_model_name(lang)
        self.tokenizer = spacy.load(pre_trained_model_name)
        self._model_name_ = pre_trained_model_name

    def tokenize(self, text):
        return [token.text for token in self.tokenizer(text)]

    @staticmethod
    def detokenize(tokenized_list):
        # TODO work on this
        return " ".join(tokenized_list)

    @property
    def model_name(self):
        return "Spacy"

    def overwrite_tokenizer_with_split_tokenizer(self):
        self.tokenizer.tokenizer = Tokenizer(self.tokenizer.vocab)

    @staticmethod
    def get_default_model_name(lang):
        if lang == "en":
            return "en_core_web_lg"
        elif lang == "de":
            return "de_core_news_md"
        elif lang == "fr":
            return "fr_core_news_md"
        elif lang == "es":
            return "es_core_news_md"
        elif lang == "pt":
            return "pt_core_news_sm"
        elif lang == "it":
            return "it_core_news_sm"
        elif lang == "nl":
            return "nl_core_news_sm"
        elif lang == "el":
            return "el_core_news_md"
        elif lang == "lt":
            return "lt_core_news_sm"
        else:
            raise ValueError("No pre-trained spacy tokenizer found for language {}".format(lang))


def get_tokenizer_from_configs(tokenizer_name, lang, lowercase_data, debug_mode=False):
    """
    A stand-alone function which will create and return the proper tokenizer object given requested configs
    """
    print("Loading tokenizer of type {} for {} language".format(tokenizer_name, lang))
    if tokenizer_name == "moses":
        return PyMosesTokenizer(lang, lowercase_data)
    elif tokenizer_name == "generic" or bool(debug_mode):
        return GenericTokenizer()
    elif tokenizer_name == "pre_trained":
        return PreTrainedTokenizer(lang, lowercase=lowercase_data)
    elif tokenizer_name == "bert":
        return PTBertTokenizer(lang, lowercase=lowercase_data)
    else:
        raise ValueError("The requested tokenizer {} does not exist or is not implemented!".format(tokenizer_name))
