import os
import io
from collections import namedtuple
from torchtext import data


class _BiAddress:
    def __init__(self):
        self.src = ''
        self.tgt = ''


class FileAddress:
    def __init__(self):
        self.train = _BiAddress()
        self.val = _BiAddress()
        self.tests = _BiAddress()


class ProcessedData:
    def __init__(self):
        self.train = None
        self.val = None
        self.test_list = None
        self.addresses = FileAddress()


class TranslationDataset(data.Dataset):
    """
    Redefines a dataset for machine translation.
    The file is copied from torchtext and modified to provide dataset related nice torchtext features,
     as well as flexibility to augment the reader with custom settings. The modified class loads a list of test sets instead of just one.
    """

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            sentence_count_limit = -1
            if "sentence_count_limit" in kwargs and kwargs["sentence_count_limit"] != -1:
                sentence_count_limit = kwargs["sentence_count_limit"] + 1
                del kwargs['sentence_count_limit']
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))
                sentence_count_limit -= 1
                if not sentence_count_limit:
                    break

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, exts, fields, path=None, root='.data',
               train='train', validation='val', test_list=('test',), **kwargs):
        """Create dataset objects for splits of a TranslationDataset.
        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test_list: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        debug_mode = False
        if "debug_mode" in kwargs:
            debug_mode = kwargs["debug_mode"]
            del kwargs["debug_mode"]
        if 'path' not in kwargs and path is None:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        elif path is None:
            path = kwargs['path']
            del kwargs['path']
        if path is None:
            path = cls.download(root)
        elif not os.path.exists(path):
            path = cls.download(root, check=path)
        if train is not None:
            if not os.path.exists(os.path.join(path, train) + exts[0]):
                print("cleaning path data ...")
                cls.clean(path)
            print("    [torchtext] Loading train examples ...")
        train_data = None if train is None else cls(os.path.join(path, train), exts, fields, **kwargs)
        if "filter_pred" in kwargs and not debug_mode:
            del kwargs['filter_pred']
        if "sentence_count_limit" in kwargs:
            del kwargs['sentence_count_limit']
        print("    [torchtext] Loading validation examples ...")
        val_data = None if validation is None else cls(os.path.join(path, validation), exts, fields, **kwargs)
        val_data.name = validation
        print("    [torchtext] Loading test examples ...")
        test_data_list = [None if test is None else cls(os.path.join(path, test), exts, fields, **kwargs) for test in test_list]
        if len(test_list):
            for d, n in zip(test_data_list, test_list):
                d.name = n
        return tuple(d for d in (train_data, val_data, *test_data_list)
                     if d is not None)

    @staticmethod
    def clean(path):
        return

    @staticmethod
    def prepare_dataset(src_lan: str, tgt_lan: str, SRC: data.Field, TGT: data.Field, load_train_data: bool, max_sequence_length: int = -1,
                        sentence_count_limit: int = -1, debug_mode: bool = False) -> ProcessedData:
        raise NotImplementedError
