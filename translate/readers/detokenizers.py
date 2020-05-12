"""
This file is intended for detokeniztion of Neural Machine Translation output results
"""
from sacremoses import MosesDetokenizer
from configuration import cfg

detokenizer = MosesDetokenizer(lang=cfg.tgt_lang)

replacement_pairs = [("& apos;", "\'"), (" / ", "/"), ("a. m.", "a.m."), ("p. m.", "p.m."), ("& amp;", "&"),
                     ("u. s. a.", "u.s.a."), ("i. e.", "i.e."), ("& quot;", "\""), ("& # 91; ", "["),
                     (" & # 93;", "]"), (" & # 124; ", "|")]


def _modify_hyphen_quote_apos(decoded, dp):
    tokens = decoded.split()
    result = ""
    quote_seen = 0
    apos_seen = 0
    for t_ind, token in enumerate(tokens):
        add_space = True
        if token == "-":
            before_cnd = t_ind > 0 and tokens[t_ind - 1] not in dp.ei_before_hyphen_list
            after_cnd = t_ind < len(tokens) - 1 and tokens[t_ind + 1] not in dp.ei_after_hyphen_list
            if after_cnd and before_cnd:
                result = result.strip()
                add_space = False
            result += token
        elif token.startswith("\'"):
            cnt = decoded.count('\'')
            if t_ind < len(tokens) - 1 and tokens[t_ind + 1] in ["s", "ll", "re"]:
                add_space = False
                result = result.strip()
            elif (cnt - apos_seen) % 2 == 0:
                add_space = False
            else:
                result = result.strip()
            result += token
            apos_seen += 1
        elif token.startswith("\""):
            cnt = decoded.count('\"')
            if (cnt - quote_seen) % 2 == 0:
                add_space = False
            else:
                result = result.strip()
            result += token
            quote_seen += 1
        elif token.endswith(".") and t_ind < len(tokens) - 1 and tokens[t_ind + 1] in ["com", "org", "info", "ca"]:
            # needs more url endings
            add_space = False
            result = result.strip()
            result += token
        elif token.endswith(".") and token.replace('.', '', 1).isdigit() and \
                t_ind < len(tokens) - 1 and tokens[t_ind + 1].replace('.', '', 1).isdigit():
            add_space = False
            result += token
        else:
            result += token
        if add_space:
            result += " "
    return result.strip()


def detokenize(tokenized_list, dp):
    """
    :param tokenized_list: the list of tokens which is supposed to be detokenized
    :param dp: the Dataprovider object which contains the dataset information
    :return: a unique detokenized str
    """
    decoded = detokenizer.detokenize(tokenized_list)
    if bool(cfg.dataset_is_in_bpe):
        decoded = decoded.replace("@@ ", "").replace(" @-@ ", "-")
    if cfg.tgt_tokenizer == "pre_trained":
        decoded = decoded.replace(" ##", "")
    for r_pair in replacement_pairs:
        decoded = decoded.replace(r_pair[0], r_pair[1])
    return _modify_hyphen_quote_apos(decoded, dp)
    # Naive detokenization
    # return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in result]).strip()
