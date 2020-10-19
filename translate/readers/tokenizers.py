"""
Implementation of different tokenizers to be used by the data provider. The pre-trained tokenizers are intended to use
  the pre-trained vocabulary files distributed by huggingface.tokenizers which are trained over big corpora
    in each supported language.
"""
from tokenizers import BertWordPieceTokenizer
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer
from requests import get
import spacy
from spacy.tokenizer import Tokenizer
import os
import string

try:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    from transformers import BertTokenizer
except ImportError:
    warnings.warn("transformers package is not available, transformers.BertTokenizer will not be accessible.")
    BertTokenizer = None


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
        "moses-pre-tokenized-wmt-uncased-fr": "https://drive.google.com/uc?export=download&id=1kYxOhJh4UshVE_SGYMANjLn_oEB6RMYC",
        "moses-pre-tokenized-wmt-uncased-en": "https://drive.google.com/uc?export=download&id=1hIURG9eiIXQYCm8cS4vJM3RLVl6UcW32",
        "moses-pre-tokenized-paracrawl-uncased-accented-de": "https://drive.google.com/uc?export=download&id=15EKdo2IXyyfZvrpOEwtx4KgeeL6Ot-Gi"
    }

    def __init__(self, lang, root='../.data', clean_text=False, handle_chinese_chars=True, strip_accents=False, lowercase=True, is_src=True):
        """
        Example instantiation: PreTrainedTokenizer("bert-base-uncased", root="../.data")
        """
        pre_trained_model_name = self.get_default_model_name(lang, lowercase, is_src)
        self._model_name_ = pre_trained_model_name
        if not os.path.exists(root):
            os.mkdir(root)
        assert pre_trained_model_name in self.vocab_files, \
            "The requested pre_trained tokenizer model {} does not exist!".format(pre_trained_model_name)
        url = self.vocab_files[pre_trained_model_name]
        f_name = root + "/" + pre_trained_model_name + ".txt"
        if not os.path.exists(f_name):
            with open(f_name, "wb") as file_:
                response = get(url)
                file_.write(response.content)
        self.moses_tkn = PyMosesTokenizer(lang, lowercase)
        self.tokenizer = BertWordPieceTokenizer(f_name, clean_text=clean_text, lowercase=lowercase,
                                                handle_chinese_chars=handle_chinese_chars, strip_accents=strip_accents)
        self.mid_tokens = {".": "&middot;", "-": "&hyphen;", "\'": "&midapos;", ",": "&midcma;", " ": "&finspace;"}
        self.reverse_mid_tokens = {v: k for k, v in self.mid_tokens.items()}
        self.lang = lang

    def get_tokenized_sub_tokens(self, token, mid_sign):
        result = []
        if mid_sign in self.mid_tokens:
            sub_tokens = token.split(mid_sign)
            assert len(sub_tokens) > 1
            if not len("".join(sub_tokens)):
                result.append(self.mid_tokens[" "])
            for sub_token in sub_tokens[:-1]:
                result.append(sub_token)
                result.append(self.mid_tokens[mid_sign])
            if len(sub_tokens[-1]) or (mid_sign == '\'' and self.lang == "fr"):
                result.append(sub_tokens[-1])
            else:  # case like "p.m." where the last token is empty
                result.append(self.mid_tokens[" "])
        else:
            result.append(token)
        return result

    def tokenize_token(self, tokens, mid_sign):
        res = []
        for token in tokens:
            if len(token) > 1 and mid_sign in token:
                for sub_token in self.get_tokenized_sub_tokens(token, mid_sign):
                    res.append(sub_token)
            else:
                res.append(token)
        return res

    def tokenize(self, text):
        """
        You can recover the output of this function using " ".join(encoded_list).replace(" ##", "")
        :param text: one line of text in type of str
        :return a list of tokenized "str"s
        """
        if not len(text.strip()):
            return [""]
        tokens = []
        for token in self.moses_tkn.tokenize(text):
            if token.startswith("&apos;") and token != "&apos;":
                token = token.replace("&apos;", "\'")
            if self.lang == "fr" and len(token) > 1 and token[1:] == "&apos;":
                token = token.replace("&apos;", "\'")
            elif self.lang == "fr" and "qu&apos;" in token:
                token = token.replace("&apos;", "\'")
            sub_ts = [token]
            for mid_sign in self.mid_tokens:
                sub_ts = self.tokenize_token(sub_ts, mid_sign)
            for sub_token in sub_ts:
                tokens.append(sub_token)
        # encoding = self.tokenizer.encode(n_text, add_special_tokens=False)
        encoding = self.tokenizer.encode(tokens, is_pretokenized=True, add_special_tokens=False)
        # encoding contains "ids", "tokens", and "offsets"
        return encoding.tokens

    def detokenize(self, tokenized_list):
        # TODO make it work on more test examples
        temp_result = []
        # Merging sub-tokens
        for token in tokenized_list:
            if len(temp_result) and token.startswith("##"):
                temp_result[-1] = temp_result[-1] + token[2:]
            else:
                temp_result.append(token)
        result = []
        index = 0
        t_len = len(temp_result)
        # merging & tokens for moses decoder
        while index < t_len:
            if temp_result[index] == "&" and index < t_len - 2 and temp_result[index + 2] == ";":
                result.append("".join(temp_result[index:index+3]))
                index += 3
            elif temp_result[index] == "&" and index < t_len - 3 and temp_result[index + 3] == ";":
                result.append("".join(temp_result[index:index+4]))
                index += 4
            else:
                result.append(temp_result[index])
                index += 1
        del temp_result[:]
        index = 0
        t_len = len(result)
        # merging &hyphen; tokens for moses decoder
        while index < t_len:
            if result[index] in self.reverse_mid_tokens:
                if not len(temp_result):
                    temp_result.append("")
                if index + 1 < t_len and result[index+1] in self.reverse_mid_tokens:  # final dot in "p.m."
                    temp_result[-1] += self.reverse_mid_tokens[result[index]] + self.reverse_mid_tokens[result[index+1]]
                    index += 2
                elif index + 1 < t_len:  # middle dot in "p.m."
                    temp_result[-1] += self.reverse_mid_tokens[result[index]] + result[index+1]
                    index += 2
                else:  # any thing else"
                    temp_result[-1] += self.reverse_mid_tokens[result[index]]
                    index += 1
            else:
                temp_result.append(result[index])
                index += 1
        return self.moses_tkn.detokenize(temp_result)

    def decode(self, encoded_ids_list):
        """
        :param encoded_ids_list: list of int ids
        :return a decoded str
        """
        decoded = self.tokenizer.decode(encoded_ids_list)
        return decoded

    @staticmethod
    def get_default_model_name(lang, lowercase, is_src=True):
        if lang == "en" and lowercase:
            return "bert-base-uncased"
        elif lang == "en" and not lowercase:
            return "bert-base-cased"
        elif lang == "zh":
            return "bert-base-chinese"
        elif lang == "de" and lowercase and not is_src:
            return "moses-pre-tokenized-paracrawl-uncased-accented-de"
        elif lang == "de" and lowercase:
            return "bert-base-german-dbmdz-uncased"
        elif lang == "de" and not lowercase:
            return "bert-base-german-dbmdz-cased"
        elif lang == "fi" and lowercase:
            return "bert-base-finnish-uncased-v1"
        elif lang == "fi" and not lowercase:
            return "bert-base-finnish-cased-v1"
        elif lang == "fr" and lowercase:
            return "moses-pre-tokenized-wmt-uncased-fr"
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
        self.lang = lang

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
        f_result = self.detokenizer.detokenize(temp_result.strip().split())
        if len(f_result) > 3 and f_result[-3] in string.punctuation and f_result[-2] == " " and f_result[-1] == "\"":
            f_result = f_result[:-2] + f_result[-1]
        return f_result

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
    def __init__(self, lang, lowercase):
        pre_trained_model_name = self.get_default_model_name(lang)
        self.tokenizer = spacy.load(pre_trained_model_name)
        self._model_name_ = pre_trained_model_name
        self.lowercase = lowercase

    def tokenize(self, text):
        return [token.text for token in self.tokenizer(text.lower() if self.lowercase else text)]

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


def get_tokenizer_from_configs(tokenizer_name, lang, lowercase_data, debug_mode=False, is_src=True):
    """
    A stand-alone function which will create and return the proper tokenizer object given requested configs
    """
    print("Loading tokenizer of type {} for {} language".format(tokenizer_name, lang))
    if tokenizer_name == "moses":
        return PyMosesTokenizer(lang, lowercase_data)
    elif tokenizer_name == "generic" or bool(debug_mode):
        return GenericTokenizer()
    elif tokenizer_name == "pre_trained":
        return PreTrainedTokenizer(lang, lowercase=lowercase_data, is_src=is_src)
    elif tokenizer_name == "spacy":
        return SpacyTokenizer(lang, lowercase=lowercase_data)
    elif tokenizer_name == "bert":
        return PTBertTokenizer(lang, lowercase=lowercase_data)
    else:
        raise ValueError("The requested tokenizer {} does not exist or is not implemented!".format(tokenizer_name))
