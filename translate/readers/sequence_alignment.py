import unidecode


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
        try:
            print("Warning one sentence is not aligned properly:\n{}\n{}\n{}\n{}".format(
                sequence_1, sequence_2, sum(fertilities), len(sequence_2)))
        except UnicodeEncodeError:
            outp = "Warning one sentence is not aligned properly:\n{}\n{}\n{}\n{}".format(
                sequence_1, sequence_2, sum(fertilities), len(sequence_2))
            print(outp.encode('ascii', 'ignore'))
    return fertilities
