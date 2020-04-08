from torchtext import data, datasets
from configuration import cfg, device
from readers.dataset import IWSLT, WMT19DeEn
import unidecode
from collections import Counter
from tqdm import tqdm


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class MyIterator(data.Iterator):
    """
    The customized torchtext iterator suggested in https://nlp.seas.harvard.edu/2018/04/03/attention.html
    The iterator is meant to speed up the training by token-wise batching
    """
    def __len__(self):
        return 0.0

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


def get_dataset(src_lan, tgt_lan, SRC: data.Field, TGT: data.Field, dev_data=None, test_data=None):
    if cfg.dataset_name == "multi30k16":
        print("Loading Multi30k [MinLen:1;AvgLen:12;MaxLen:40]")
        train, val, test = datasets.translation.Multi30k.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
                                                                fields=(SRC, TGT))
        src_val_file_address = ".data/multi30k/val.{}".format(src_lan)
        tgt_val_file_address = ".data/multi30k/val.{}".format(tgt_lan)
        src_test_file_address = ".data/multi30k/test2016.{}".format(src_lan)
        tgt_test_file_address = ".data/multi30k/test2016.{}".format(tgt_lan)
        src_train_file_address = ".data/multi30k/train.{}".format(src_lan)
        tgt_train_file_address = ".data/multi30k/train.{}".format(tgt_lan)
    elif cfg.dataset_name == "iwslt17":
        dev_data = dev_data if dev_data is not None else "dev2010"
        test_data = test_data if test_data is not None else "tst2015"
        train, val, test = IWSLT.splits(
            filter_pred=lambda x: len(vars(x)['src']) <= cfg.max_sequence_length and len(
                vars(x)['trg']) <= cfg.max_sequence_length, exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
            fields=(SRC, TGT), test='IWSLT17.TED.{}'.format(test_data), validation='IWSLT17.TED.{}'.format(dev_data),
            debug_mode=bool(cfg.debug_mode))
        # TODO this will break if we reverse the src and target since .data/iwslt/en-de/ does not exist
        src_val_file_address = ".data/iwslt/{0}-{1}/IWSLT17.TED.{2}.{0}-{1}.{0}".format(src_lan, tgt_lan, dev_data)
        tgt_val_file_address = ".data/iwslt/{0}-{1}/IWSLT17.TED.{2}.{0}-{1}.{1}".format(src_lan, tgt_lan, dev_data)
        src_test_file_address = ".data/iwslt/{0}-{1}/IWSLT17.TED.{2}.{0}-{1}.{0}".format(src_lan, tgt_lan, test_data)
        tgt_test_file_address = ".data/iwslt/{0}-{1}/IWSLT17.TED.{2}.{0}-{1}.{1}".format(src_lan, tgt_lan, test_data)
        src_train_file_address = ".data/iwslt/{0}-{1}/train.{0}-{1}.{0}".format(src_lan, tgt_lan)
        tgt_train_file_address = ".data/iwslt/{0}-{1}/train.{0}-{1}.{1}".format(src_lan, tgt_lan)
    elif cfg.dataset_name == "wmt14":
        dev_data = dev_data if dev_data is not None else "newstest2009"
        test_data = test_data if test_data is not None else "newstest2016"
        train, val, test = datasets.WMT14.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
                                                 fields=(SRC, TGT), train='train.tok.clean.bpe.32000',
                                                 validation='{}.tok.bpe.32000'.format(dev_data),
                                                 test='{}.tok.bpe.32000'.format(test_data))
        src_val_file_address = ".data/wmt14/{}.tok.bpe.32000.{}".format(dev_data, src_lan)
        tgt_val_file_address = ".data/wmt14/{}.tok.bpe.32000.{}".format(dev_data, tgt_lan)
        src_test_file_address = ".data/wmt14/{}.tok.bpe.32000.{}".format(test_data, src_lan)
        tgt_test_file_address = ".data/wmt14/{}.tok.bpe.32000.{}".format(test_data, tgt_lan)
        src_train_file_address = ".data/wmt14/train.tok.clean.bpe.32000.{}".format(src_lan)
        tgt_train_file_address = ".data/wmt14/train.tok.clean.bpe.32000.{}".format(tgt_lan)
    elif cfg.dataset_name == "wmt19_de_en_sample":
        dev_data = dev_data if dev_data is not None else "newstest2018"
        test_data = test_data if test_data is not None else "newstest2019"
        train_data = 'train.samll'
        train, val, test = WMT19DeEn.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
                                            fields=(SRC, TGT), train=train_data,
                                            validation='{}-deen'.format(dev_data),
                                            test='{}-deen'.format(test_data))
        src_val_file_address = ".data/wmt19_en_de/{}-deen.{}".format(dev_data, src_lan)
        tgt_val_file_address = ".data/wmt19_en_de/{}-deen.{}".format(dev_data, tgt_lan)
        src_test_file_address = ".data/wmt19_en_de/{}-deen.{}".format(test_data, src_lan)
        tgt_test_file_address = ".data/wmt19_en_de/{}-deen.{}".format(test_data, tgt_lan)
        src_train_file_address = ".data/wmt19_en_de/{}.{}".format(train_data, src_lan)
        tgt_train_file_address = ".data/wmt19_en_de/{}.{}".format(train_data, tgt_lan)
    else:
        raise ValueError("The dataset {} is not defined in torchtext or SFUTranslate!".format(cfg.dataset_name))

    return train, val, test, src_val_file_address, tgt_val_file_address, src_test_file_address, tgt_test_file_address, \
        src_train_file_address, tgt_train_file_address


def check_for_inital_subword_sequence(sequence_1, sequence_2):
    seg_s_i = 0
    sequence_1_token = sequence_1[seg_s_i]
    sequence_2_f_pointer = 0
    sequence_2_token = sequence_2[sequence_2_f_pointer]
    while not check_tokens_equal(sequence_1_token, sequence_2_token):
        sequence_2_f_pointer += 1
        if sequence_2_f_pointer < len(sequence_2):
            tmp = sequence_2[sequence_2_f_pointer]
        else:
            sequence_2_f_pointer = 0
            seg_s_i = -1
            break
        sequence_2_token += tmp[2:] if tmp.startswith("##") else tmp
    return seg_s_i, sequence_2_f_pointer


def check_tokens_equal(sequence_1_token, sequence_2_token):
    if sequence_1_token is None:
        return sequence_2_token is None
    if sequence_2_token is None:
        return sequence_1_token is None
    if sequence_2_token == sequence_1_token.lower() or sequence_2_token == sequence_1_token:
        return True
    sequence_1_token = unidecode.unidecode(sequence_1_token)  # remove accents and unicode emoticons
    sequence_2_token = unidecode.unidecode(sequence_2_token)  # remove accents and unicode emoticons
    if sequence_2_token == sequence_1_token.lower() or sequence_2_token == sequence_1_token:
        return True
    return False


def find_token_index_in_list(sequence_1, tokens_doc, check_lowercased_doc_tokens=False):
    if sequence_1 is None or tokens_doc is None or not len(tokens_doc):
        return []
    if check_lowercased_doc_tokens:
        inds = [i for i, val in enumerate(tokens_doc) if check_tokens_equal(
            sequence_1, val) or check_tokens_equal(sequence_1, val.lower())]
    else:
        inds = [i for i, val in enumerate(tokens_doc) if check_tokens_equal(sequence_1, val)]
    # assert len(inds) == 1
    return inds


def extract_monotonic_sequence_to_sequence_alignment(sequence_1, sequence_2, print_alignments=False, level=0):
    """
    This function receives two lists of string tokens expected to be monotonically pseudo-aligned,
      and returns the alignment fertility values from :param sequence_1: to :param sequence_2:.
    The output will have a length equal to the size of :param sequence_1: each index of which indicates the number
      of times the :param sequence_1: element must be copied to equal the length of :param sequence_2: list.
    This algorithm enforces the alignments in a strictly left-to-right order.
    This algorithm is mainly designed for aligning the outputs of two different tokenizers (e.g. bert and spacy)
      on the same input sentence.
    """
    previous_sequence_1_token = None
    sp_len = len(sequence_1)
    bt_len = len(sequence_2)
    if not sp_len:
        return []
    elif not bt_len:
        return [0] * len(sequence_1)
    elif sp_len == 1:  # one to one and one to many
        return [bt_len]
    elif bt_len == 1:  # many to one case
        r = [0] * sp_len
        r[0] = 1
        return r
    # many to many case is being handled in here:
    seg_s_i = -1
    seg_sequence_2_f_pointer = -1
    best_right_candidate = None
    best_left_candidate = None
    for s_i in range(sp_len):
        sequence_1_token = sequence_1[s_i]
        next_sequence_1_token = sequence_1[s_i + 1] if s_i < len(sequence_1) - 1 else None
        prev_eq = None
        current_eq = None
        next_eq = None
        previous_sequence_2_token = None
        exact_expected_location_range_list = find_token_index_in_list(sequence_1_token, sequence_2)
        if not len(exact_expected_location_range_list):
            exact_expected_location_range = -1
        elif len(exact_expected_location_range_list) == 1:
            exact_expected_location_range = exact_expected_location_range_list[0]
        else:  # multiple options to choose from
            selection_index_list = find_token_index_in_list(sequence_1_token, sequence_1, check_lowercased_doc_tokens=True)
            # In cases like [hadn 't and had n't] or wrong [UNK] merges:
            #       len(exact_expected_location_range_list) < len(selection_index_list)
            # In cases like punctuations which will get separated in s2 tokenizer and don't in s1 or subword breaks
            #       len(exact_expected_location_range_list) > len(selection_index_list)
            selection_index = selection_index_list.index(s_i)
            if selection_index < len(exact_expected_location_range_list):
                # TODO account for distortion (if some other option has less distortion take it)
                exact_expected_location_range = exact_expected_location_range_list[selection_index]
            else:
                # raise ValueError("selection_index is greater than the available list")
                # TODO obviously not the best choice but I have to select something after all
                exact_expected_location_range = exact_expected_location_range_list[-1]
        end_of_expected_location_range = exact_expected_location_range+1 if exact_expected_location_range > -1 else s_i+len(sequence_1_token)+2
        start_of_expected_location_range = exact_expected_location_range - 1 if exact_expected_location_range > -1 else s_i-1
        for sequence_2_f_pointer in range(
                max(start_of_expected_location_range, 0), min(len(sequence_2), end_of_expected_location_range)):
            sequence_2_token = sequence_2[sequence_2_f_pointer]
            next_sequence_2_token = sequence_2[sequence_2_f_pointer + 1] if sequence_2_f_pointer < len(sequence_2) - 1 else None
            prev_eq = check_tokens_equal(previous_sequence_1_token, previous_sequence_2_token)
            current_eq = check_tokens_equal(sequence_1_token, sequence_2_token)
            next_eq = check_tokens_equal(next_sequence_1_token, next_sequence_2_token)
            if prev_eq and current_eq and next_eq:
                seg_sequence_2_f_pointer = sequence_2_f_pointer
                break
            elif prev_eq and current_eq and best_left_candidate is None:
                best_left_candidate = (s_i, sequence_2_f_pointer)
            elif current_eq and next_eq and best_right_candidate is None:
                best_right_candidate = (s_i, sequence_2_f_pointer)
            previous_sequence_2_token = sequence_2_token
        if prev_eq and current_eq and next_eq:
            seg_s_i = s_i
            break
        previous_sequence_1_token = sequence_1_token

    curr_fertilities = [1]
    if seg_s_i == -1 or seg_sequence_2_f_pointer == -1:
        if best_left_candidate is not None and best_right_candidate is not None:  # accounting for min distortion
            seg_s_i_l, seg_sequence_2_f_pointer_l = best_left_candidate
            seg_s_i_r, seg_sequence_2_f_pointer_r = best_right_candidate
            if seg_sequence_2_f_pointer_r - seg_s_i_r < seg_sequence_2_f_pointer_l - seg_s_i_l:
                seg_s_i, seg_sequence_2_f_pointer = best_right_candidate
            else:
                seg_s_i, seg_sequence_2_f_pointer = best_left_candidate
        elif best_left_candidate is not None:
            seg_s_i, seg_sequence_2_f_pointer = best_left_candidate
        elif best_right_candidate is not None:
            seg_s_i, seg_sequence_2_f_pointer = best_right_candidate
        else:  # multiple subworded tokens stuck together
            seg_s_i, seg_sequence_2_f_pointer = check_for_inital_subword_sequence(sequence_1, sequence_2)
            curr_fertilities = [seg_sequence_2_f_pointer + 1]
    if seg_s_i == -1 or seg_sequence_2_f_pointer == -1 and len(sequence_1[0]) < len(sequence_2[0]): # none identical tokenization
        seg_s_i = 0
        seg_sequence_2_f_pointer = 0
    if seg_s_i == -1 or seg_sequence_2_f_pointer == -1:
        print(sequence_1)
        print(sequence_2)
        raise ValueError()
    if seg_s_i > 0:  # seg_sequence_2_f_pointer  is always in the correct range
        left = extract_monotonic_sequence_to_sequence_alignment(
            sequence_1[:seg_s_i], sequence_2[:seg_sequence_2_f_pointer], False, level + 1)
    else:
        left = []
    if seg_s_i < sp_len:  # seg_sequence_2_f_pointer  is always in the correct range
        right = extract_monotonic_sequence_to_sequence_alignment(
            sequence_1[seg_s_i + 1:], sequence_2[seg_sequence_2_f_pointer + 1:], False, level + 1)
    else:
        right = []
    fertilities = left + curr_fertilities + right
    if print_alignments and not level:
        sequence_2_ind = 0
        for src_token, fertility in zip(sequence_1, fertilities):
            for b_f in range(fertility):
                print("{} --> {}".format(src_token, sequence_2[sequence_2_ind + b_f]))
            sequence_2_ind += fertility
    if not level and sum(fertilities) != len(sequence_2):
        print("Warning one sentence is not aligned properly:\n{}\n{}\n{}\n{}".format(
            sequence_1, sequence_2, sum(fertilities), len(sequence_2)))
    return fertilities


def collect_unk_stats(SRC, TGT, src_tokenizer, tgt_tokenizer, dt_raw, dataset_name,
                      src_file_adr, tgt_file_adr, src_unk_token, m_unk_token="\u26F6"):
    from utils.evaluation import convert_target_batch_back
    dt_iter = data.BucketIterator(dt_raw, batch_size=1, device=device, repeat=False, train=False, shuffle=False,
                                  sort=False, sort_within_batch=False)
    to_lower = bool(cfg.lowercase_data)
    src_originals = iter(open(src_file_adr, "r"))
    tgt_originals = iter(open(tgt_file_adr, "r"))
    src_unk_cnt = Counter()
    trg_unk_cnt = Counter()
    src_cnt = Counter()
    trg_cnt = Counter()
    print("Collecting UNK token/type statistics ...")
    for dt, s, t in tqdm(zip(dt_iter, src_originals, tgt_originals)):
        dt_src_proc = convert_target_batch_back(dt.src[0], SRC)[0].replace(cfg.unk_token, m_unk_token).replace(" ##","")
        dt_src_orig = s.lower().strip() if to_lower else s.strip()
        dt_trg_proc = convert_target_batch_back(dt.trg[0], TGT)[0].replace(cfg.unk_token, m_unk_token).replace(" ##","")
        dt_trg_orig = t.lower().strip() if to_lower else t.strip()
        dt_src = src_tokenizer(dt_src_proc)
        dt_trg = tgt_tokenizer(dt_trg_proc)
        st = src_tokenizer(dt_src_orig)
        tt = tgt_tokenizer(dt_trg_orig)
        src_a = extract_monotonic_sequence_to_sequence_alignment(dt_src, st)
        trg_a = extract_monotonic_sequence_to_sequence_alignment(dt_trg, tt)
        ss_idx = 0
        for s_token, f in zip(dt_src, src_a):
            for i in range(f):
                ss_token = st[ss_idx+i]  # st will never contain [UNK] token.
                if src_unk_token == s_token:
                    src_unk_cnt[ss_token] += 1.0
                src_cnt[ss_token] += 1.0
            ss_idx += f
        tt_idx = 0
        for t_token, f in zip(dt_trg, trg_a):
            for i in range(f):
                tt_token = tt[tt_idx+i]  # tt will never contain [UNK] token.
                if m_unk_token == t_token:
                    trg_unk_cnt[tt_token] += 1.0
                trg_cnt[tt_token] += 1.0
            tt_idx += f

    print("UNK percentage of source vocabulary types over {} data: {:.2f}% ".format(dataset_name, float(
        len(src_unk_cnt)) * 100.0 / len(src_cnt)))
    print("UNK percentage of target vocabulary types over {} data: {:.2f}% ".format(dataset_name, float(
        len(trg_unk_cnt)) * 100.0 / len(trg_cnt)))
    print("UNK percentage of source vocabulary tokens over {} data: {:.2f}% ".format(dataset_name, float(
        sum(src_unk_cnt.values())) * 100.0 / sum(src_cnt.values())))
    print("UNK percentage of target vocabulary tokens over {} data: {:.2f}% ".format(dataset_name, float(
        sum(trg_unk_cnt.values())) * 100.0 / sum(trg_cnt.values())))
