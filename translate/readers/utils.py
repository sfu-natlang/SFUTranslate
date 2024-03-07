from configuration import cfg, device
from collections import Counter
from tqdm import tqdm
from readers.iterators import MyBucketIterator


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


def collect_unk_stats(SRC, TGT, src_tokenizer, tgt_tokenizer, dt_raw, dataset_name,
                      src_file_adr, tgt_file_adr, src_unk_token, m_unk_token="\u26F6"):
    from utils.evaluation import convert_target_batch_back
    from readers.sequence_alignment import extract_monotonic_sequence_to_sequence_alignment

    def _get_next_line(file_1_iter, file_2_iter):
        for l1, l2 in zip(file_1_iter, file_2_iter):
            l1 = l1.strip()
            l2 = l2.strip()
            if not len(l1) or not len(l2):
                continue
            yield l1, l2
    dt_iter = MyBucketIterator(dt_raw, batch_size=1, device=device, repeat=False, train=False, shuffle=False,
                                  sort=False, sort_within_batch=False)
    # BucketIterator will ignore the lines one side of which is an empty line
    to_lower = bool(cfg.lowercase_data)
    src_originals = iter(open(src_file_adr, "r"))
    tgt_originals = iter(open(tgt_file_adr, "r"))
    src_unk_cnt = Counter()
    trg_unk_cnt = Counter()
    src_cnt = Counter()
    trg_cnt = Counter()
    print("Collecting UNK token/type statistics ...")
    for dt, (s, t) in tqdm(zip(dt_iter, _get_next_line(src_originals, tgt_originals))):
        dt_src_proc = convert_target_batch_back(dt.src[0], SRC)[0].replace(cfg.unk_token, m_unk_token).replace(" ##","")
        dt_src_orig = s.lower() if to_lower else s
        dt_trg_proc = convert_target_batch_back(dt.trg[0], TGT)[0].replace(cfg.unk_token, m_unk_token).replace(" ##","")
        dt_trg_orig = t.lower() if to_lower else t
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


def extract_exclusive_immediate_neighbours(data_file, dtoken="-", to_lower=False, consider_intersect_as_well=False):
    """
    given a :param data_file: , this function looks through the tokens of each line and extracts the list of tokens that
      have strictly been seen before or after :param dtoken: but not in a single token with it.
    """
    info = {"inside": {"before": set(), "after": set()}, "separate": {"before": set(), "after": set()}}
    for f_line in open(data_file, "r", encoding="utf-8"):
        if to_lower:  # recover the original sentence
            f_line = f_line.lower()
        if dtoken not in f_line:
            continue
        ls = f_line.split()
        for tind, t in enumerate(ls):  # look at each of the tokens
            if dtoken == t:
                if tind > 0:
                    info["separate"]["before"].add(ls[tind-1])
                if tind < len(ls)-1:
                    info["separate"]["after"].add(ls[tind+1])
            elif dtoken in t:
                ws = t.split("-")
                for w_ind in range(len(ws)-1):
                    info["inside"]["before"].add(ws[w_ind])
                    info["inside"]["after"].add(ws[w_ind+1])
    if consider_intersect_as_well:
        befores = [k for k in info["separate"]["before"]]
        afters = [k for k in info["separate"]["after"]]
    else:
        befores = [k for k in info["separate"]["before"] if k not in info["inside"]["before"]]
        afters = [k for k in info["separate"]["after"] if k not in info["inside"]["after"]]
    return befores, afters
