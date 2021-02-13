"""
The modified implementations of torchtext.datasets classes containing updated dataset urls as well as extension checking and a unified splits function
"""
import os
import io
import glob
import codecs
import xml.etree.ElementTree as ET

from readers.datasets.generic import TranslationDataset, ProcessedData


class M30k(TranslationDataset):
    """The small-dataset in WMT 2017 multimodal task [MinLen:1;AvgLen:12;MaxLen:40]"""

    urls = [('https://drive.google.com/uc?export=download&'
             'id=1qJEyZnF6heNvcJtw-t3YbQ4U_t8IJSGb', 'M30k2018.zip')]
    name = 'm30k'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data', train='train', validation='val',
               test_list=('test_2016_flickr', 'test_2017_flickr', 'test_2018_flickr', 'test_2017_mscoco'), **kwargs):
        if exts[0][1:] not in ['fr', 'en', 'de'] or exts[1][1:] not in ['fr', 'en', 'de']:
            raise ValueError("This data set only contains data translated to/from English, French, or German")
        return super(M30k, cls).splits(exts, fields, None, root, train, validation, test_list, **kwargs)

    @staticmethod
    def prepare_dataset(root, src_lan, tgt_lan, SRC, TGT, load_train_data, max_sequence_length, sentence_count_limit, debug_mode) -> ProcessedData:
        res = ProcessedData()
        test_data_list = ['test_2016_flickr', 'test_2017_flickr', 'test_2018_flickr', 'test_2017_mscoco']
        print("Loading Multi30k dataset ...")
        if load_train_data:
            train, val, *test = M30k.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)), fields=(SRC, TGT),
                                            sentence_count_limit=sentence_count_limit, root=root)
        else:
            val, *test = M30k.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)), fields=(SRC, TGT), train=None,
                                     sentence_count_limit=sentence_count_limit, root=root)
            train = None
        val.name = "multi30k.dev"
        res.train = train
        res.val = val
        res.test_list = test
        res.target_language = tgt_lan
        res.addresses.val.src = "{}/m30k/val.{}".format(root, src_lan)
        res.addresses.val.tgt = "{}/m30k/val.{}".format(root, tgt_lan)
        res.addresses.tests.src = ["{}/m30k/{}.{}".format(root, d_set, src_lan) for d_set in test_data_list]
        res.addresses.tests.tgt = ["{}/m30k/{}.{}".format(root, d_set, tgt_lan) for d_set in test_data_list]
        res.addresses.val.src_sgm = "{}/m30k/val.{}-{}.{}.sgm".format(root, src_lan, tgt_lan, src_lan)
        res.addresses.val.tgt_sgm = "{}/m30k/val.{}-{}.{}.sgm".format(root, src_lan, tgt_lan, tgt_lan)
        res.addresses.tests.src_sgm = ["{}/m30k/{}.{}-{}.{}.sgm".format(root, d_set, src_lan, tgt_lan, src_lan) for d_set in test_data_list]
        res.addresses.tests.tgt_sgm = ["{}/m30k/{}.{}-{}.{}.sgm".format(root, d_set, src_lan, tgt_lan, tgt_lan) for d_set in test_data_list]
        res.addresses.train.src = "{}/m30k/train.{}".format(root, src_lan)
        res.addresses.train.tgt = "{}/m30k/train.{}".format(root, tgt_lan)
        return res


class IWSLT(TranslationDataset):
    """
    Do-over of the original Torchtext IWSLT Library.
    This one does not apply filter_pred designed for length limitation to the test and dev datasets.
    The IWSLT 2017 TED talk translation task - https://wit3.fbk.eu/mt.php?release=2017-01-trnted
    """

    # base_url = 'https://wit3.fbk.eu/archive/2016-01//texts/{}/{}/{}.tgz'
    base_urls = {
        'ar-en': ('https://drive.google.com/uc?export=download&id=1W-dFXLwObUWAdsPbDM6EKPdTYxqTA9hm', 'ar-en.tgz'),
        'de-en': ('https://drive.google.com/uc?export=download&id=1_YztpgqGo_qjv35R0K7smqY1vsF0M3Hj', 'de-en.tgz'),
        'fr-en': ('https://drive.google.com/uc?export=download&id=1Cp1Y5n4GcaLkiWOlly7IgUt7OcQCZIZJ', 'fr-en.tgz'),
        'ja-en': ('https://drive.google.com/uc?export=download&id=1MEgiBpQSYuEBYbreBoM9zNGQghDJiIF3', 'ja-en.tgz'),
        'ko-en': ('https://drive.google.com/uc?export=download&id=1RKyYCKzobtIVlJTZN7zw0HfjbVK_y4mz', 'ko-en.tgz'),
        'zh-en': ('https://drive.google.com/uc?export=download&id=11W8iy9SnBdB2QJqCvsqVy3OkMtVshN6j', 'zh-en.tgz'),
        'en-ar': ('https://drive.google.com/uc?export=download&id=1SkBW7U8z8QgpEKLCmSJyo67idrwsV1Hf', 'en-ar.tgz'),
        'en-de': ('https://drive.google.com/uc?export=download&id=19GXmBrUrwQQxNVHL3yHkAsoIvlUbxdKc', 'en-de.tgz'),
        'en-fr': ('https://drive.google.com/uc?export=download&id=1vA3oXkUZUqA8xzSU7IitAbJMF4hUmqPX', 'en-fr.tgz'),
        'en-ja': ('https://drive.google.com/uc?export=download&id=1wex32t_nbbDrmxHn0mAdNOMlk4yRpvnq', 'en-ja.tgz'),
        'en-ko': ('https://drive.google.com/uc?export=download&id=1mhuUMxFLi-TNwdAEvBaKF1RauvTBIzK4', 'en-ko.tgz'),
        'en-zh': ('https://drive.google.com/uc?export=download&id=1eYY9-MCi0GmjDwfuzibyV9UcCqnBzLOR', 'en-zh.tgz')
    }
    name = 'iwslt'
    base_dirname = '{}-{}'

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train', validation='IWSLT17.TED.dev2010',
               test_list=('IWSLT17.TED.tst2010', 'IWSLT17.TED.tst2011', 'IWSLT17.TED.tst2012', 'IWSLT17.TED.tst2013',
                          'IWSLT17.TED.tst2014', 'IWSLT17.TED.tst2015'), **kwargs):
        if exts[0][1:] != 'en' and exts[1][1:] != 'en':
            raise ValueError("This data set only contains data translated to/from English "
                             "when the other side is in [Arabic, German, French, Japanese, Korean, and Chinese]")
        if exts[0][1:] == 'en' and exts[1][1:] not in ['ar', 'de', 'fr', 'ja', 'ko', 'zh']:
            raise ValueError("This data set only contains data translated to/from English "
                             "when the other side is in [Arabic, German, French, Japanese, Korean, and Chinese]")
        if exts[1][1:] == 'en' and exts[0][1:] not in ['ar', 'de', 'fr', 'ja', 'ko', 'zh']:
            raise ValueError("This data set only contains data translated to/from English "
                             "when the other side is in [Arabic, German, French, Japanese, Korean, and Chinese]")
        cls.dirname = cls.base_dirname.format(exts[0][1:], exts[1][1:])
        cls.urls = [cls.base_urls[cls.dirname]]
        path = os.path.join(root, cls.name, cls.dirname)
        if train is not None:
            train = '.'.join([train, cls.dirname])
        validation = '.'.join([validation, cls.dirname])
        tests = []
        if test_list is not None:
            for t in test_list:
                tests.append('.'.join([t, cls.dirname]))
        return super(IWSLT, cls).splits(exts, fields, path, root, train, validation, tests, **kwargs)

    @staticmethod
    def clean(path):
        for f_xml in glob.iglob(os.path.join(path, '*.xml')):
            print(f_xml)
            f_txt = os.path.splitext(f_xml)[0]
            with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt:
                root = ET.parse(f_xml).getroot()[0]
                for doc in root.findall('doc'):
                    for e in doc.findall('seg'):
                        fd_txt.write(e.text.strip() + '\n')

        xml_tags = ['<url', '<keywords', '<talkid', '<description',
                    '<reviewer', '<translator', '<title', '<speaker']
        for f_orig in glob.iglob(os.path.join(path, 'train.tags*')):
            print(f_orig)
            f_txt = f_orig.replace('.tags', '')
            with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt, \
                    io.open(f_orig, mode='r', encoding='utf-8') as fd_orig:
                for l in fd_orig:
                    if not any(tag in l for tag in xml_tags):
                        fd_txt.write(l.strip() + '\n')

    @staticmethod
    def prepare_dataset(root, src_lan, tgt_lan, SRC, TGT, load_train_data, max_sequence_length, sentence_count_limit, debug_mode) -> ProcessedData:
        res = ProcessedData()
        if not load_train_data:
            val, *test = IWSLT.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)), fields=(SRC, TGT), debug_mode=debug_mode, train=None,
                                      sentence_count_limit=sentence_count_limit, root=root)
            train = None
        elif max_sequence_length > 1:
            train, val, *test = IWSLT.splits(filter_pred=lambda x: len(vars(x)['src']) <= max_sequence_length and
                                                                   len(vars(x)['trg']) <= max_sequence_length,
                                             exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)), fields=(SRC, TGT), debug_mode=debug_mode,
                                             sentence_count_limit=sentence_count_limit, root=root)
        else:
            train, val, *test = IWSLT.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)), fields=(SRC, TGT), debug_mode=debug_mode,
                                             sentence_count_limit=sentence_count_limit, root=root)
        res.train = train
        res.val = val
        res.test_list = test
        if tgt_lan == "en":
            res.target_language = "english"
        elif tgt_lan == "de":
            res.target_language = "german"
        elif tgt_lan == "fr":
            res.target_language = "french"
        elif tgt_lan == "ar":
            res.target_language = "arabic"
        elif tgt_lan == "ja":
            res.target_language = "japanese"
        elif tgt_lan == "ko":
            res.target_language = "korean"
        elif tgt_lan == "zh":
            res.target_language = "chinese"
        else:
            raise ValueError("target language {} not supported".format(tgt_lan))
        res.addresses.val.src = "{2}/iwslt/{0}-{1}/IWSLT17.TED.dev2010.de-en.{0}".format(src_lan, tgt_lan, root)
        res.addresses.val.tgt = "{2}/iwslt/{0}-{1}/IWSLT17.TED.dev2010.de-en.{1}".format(src_lan, tgt_lan, root)
        res.addresses.tests.src = ["{3}/iwslt/{0}-{1}/IWSLT17.TED.{2}.de-en.{0}".format(
            src_lan, tgt_lan, test_data, root) for test_data in ["tst201{}".format(i) for i in range(6)]]
        res.addresses.tests.tgt = ["{3}/iwslt/{0}-{1}/IWSLT17.TED.{2}.de-en.{1}".format(
            src_lan, tgt_lan, test_data, root) for test_data in ["tst201{}".format(i) for i in range(6)]]
        res.addresses.val.src_sgm = "{2}/iwslt/{0}-{1}/IWSLT17.TED.dev2010.de-en.{0}.xml".format(src_lan, tgt_lan, root)
        res.addresses.val.tgt_sgm = "{2}/iwslt/{0}-{1}/IWSLT17.TED.dev2010.de-en.{1}.xml".format(src_lan, tgt_lan, root)
        res.addresses.tests.src_sgm = ["{3}/iwslt/{0}-{1}/IWSLT17.TED.{2}.de-en.{0}.xml".format(
            src_lan, tgt_lan, test_data, root) for test_data in ["tst201{}".format(i) for i in range(6)]]
        res.addresses.tests.tgt_sgm = ["{3}/iwslt/{0}-{1}/IWSLT17.TED.{2}.de-en.{1}.xml".format(
            src_lan, tgt_lan, test_data, root) for test_data in ["tst201{}".format(i) for i in range(6)]]
        res.addresses.train.src = "{2}/iwslt/{0}-{1}/train.de-en.{0}".format(src_lan, tgt_lan, root)
        res.addresses.train.tgt = "{2}/iwslt/{0}-{1}/train.de-en.{1}".format(src_lan, tgt_lan, root)
        return res


class WMT19DeEn(TranslationDataset):
    """The WMT 2019 English-German dataset. The download and data preparation script is enclosed in the dataset folder"""

    urls = [('https://drive.google.com/uc?export=download&'
             'id=1miCyP1Vdoi6QsGoKSWhNgxbPVmSSs3Rt', 'wmt19_en_de_raw.zip'),
            ('https://drive.google.com/uc?export=download&'
             'id=1nTtizBqGpbNguSLq98t6A9xOamx17kUF', 'wmt20_dev_de_en_preprocessed.tgz')]
    name = 'wmt19_en_de'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data', train='train', validation='wmt_valid',
               test_list=('newstest2014', 'newstest2015', 'newstest2016', 'newstest2017', 'newstest2018'), **kwargs):
        if exts[0][1:] not in ['en', 'de'] or exts[1][1:] not in ['en', 'de']:
            raise ValueError("This data set only contains data translated from German to English or reverse")
        return super(WMT19DeEn, cls).splits(exts, fields, None, root, train, validation, test_list, **kwargs)

    @staticmethod
    def prepare_dataset(root, src_lan, tgt_lan, SRC, TGT, load_train_data, max_sequence_length, sentence_count_limit, debug_mode) -> ProcessedData:
        res = ProcessedData()
        dev_data = "wmt_valid"
        test_data_list = ["newssyscomb2009", "newstest2008", "newstest2009"] + ["newstest201{}".format(i) for i in range(0, 9)] + \
                         ["newstest2019-{}{}".format(src_lan, tgt_lan), "newstest2020-{}{}".format(src_lan, tgt_lan),
                          "newstestB2020-{}{}".format(src_lan, tgt_lan)]
        test_sgm_data_list = ["newssyscomb2009", "newstest2008", "newstest2009"] + ["newstest201{}".format(i) for i in range(0, 10)] + \
                             ["newstest2020", "newstestB2020"]
        train_data = "train"
        if not load_train_data:
            val, *test = WMT19DeEn.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)), fields=(SRC, TGT), train=None,
                                          validation="wmt_valid" if dev_data == "wmt_valid" else dev_data, test_list=test_data_list,
                                          sentence_count_limit=sentence_count_limit, root=root)
            train = None
        elif max_sequence_length > 1:
            train, val, *test = WMT19DeEn.splits(
                filter_pred=lambda x: len(vars(x)['src']) <= max_sequence_length and len(
                    vars(x)['trg']) <= max_sequence_length, exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
                fields=(SRC, TGT), train=train_data, validation="wmt_valid" if dev_data == "wmt_valid" else dev_data,
                test_list=test_data_list, sentence_count_limit=sentence_count_limit, root=root)
        else:
            train, val, *test = WMT19DeEn.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)), fields=(SRC, TGT), train=train_data,
                                                 validation="wmt_valid" if dev_data == "wmt_valid" else dev_data, test_list=test_data_list,
                                                 sentence_count_limit=sentence_count_limit, root=root)
        res.train = train
        res.val = val
        res.test_list = test
        res.target_language = tgt_lan
        if dev_data == "wmt_valid":
            res.addresses.val.src = "{}/wmt19_en_de/wmt_valid.{}".format(root, src_lan)
            res.addresses.val.tgt = "{}/wmt19_en_de/wmt_valid.{}".format(root, tgt_lan)
            res.addresses.val.src_sgm = "{}/wmt19_en_de/wmt_valid-src.{}.sgm".format(root, src_lan)
            res.addresses.val.tgt_sgm = "{}/wmt19_en_de/wmt_valid-ref.{}.sgm".format(root, tgt_lan)
        else:
            res.addresses.val.src = "{}/wmt19_en_de/{}.{}".format(root, dev_data, src_lan)
            res.addresses.val.tgt = "{}/wmt19_en_de/{}.{}".format(root, dev_data, tgt_lan)
            res.addresses.val.src_sgm = "{}/wmt19_en_de/{}-src.{}.sgm".format(root, dev_data, src_lan)
            res.addresses.val.tgt_sgm = "{}/wmt19_en_de/{}-ref.{}.sgm".format(root, dev_data, tgt_lan)
        res.addresses.tests.src = ["{}/wmt19_en_de/{}.{}".format(root, test_data, src_lan) for test_data in test_data_list]
        res.addresses.tests.tgt = ["{}/wmt19_en_de/{}.{}".format(root, test_data, tgt_lan) for test_data in test_data_list]
        res.addresses.tests.src_sgm = ["{}/wmt19_en_de/{}-src.{}.sgm".format(root, test_data, src_lan) for test_data in test_sgm_data_list]
        res.addresses.tests.tgt_sgm = ["{}/wmt19_en_de/{}-ref.{}.sgm".format(root, test_data, tgt_lan) for test_data in test_sgm_data_list]
        res.addresses.train.src = "{}/wmt19_en_de/{}.{}".format(root, train_data, src_lan)
        res.addresses.train.tgt = "{}/wmt19_en_de/{}.{}".format(root, train_data, tgt_lan)

        return res


class WMT19DeFr(TranslationDataset):
    """The WMT 2019 English-French dataset, processed using the script in
    https://drive.google.com/open?id=1-HJr69Z-Svl55xo5c2fco7QOCXeETh0H"""

    urls = [('https://drive.google.com/uc?export=download&'
             'id=1-HJr69Z-Svl55xo5c2fco7QOCXeETh0H', 'wmt19_de_fr.zip')]
    name = 'wmt19_de_fr'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data', train='train', validation='valid',
               test_list=('newstest2008-defr', 'newstest2009-defr', 'newstest2010-defr', 'newstest2011-defr',
                          'newstest2012-defr', 'newstest2013-defr', 'newstest2019-defr', 'euelections_dev2019-defr'), **kwargs):
        if exts[0][1:] not in ['fr', 'de'] or exts[1][1:] not in ['fr', 'de']:
            raise ValueError("This data set only contains data translated from German to French or reverse")
        return super(WMT19DeFr, cls).splits(exts, fields, None, root, train, validation, test_list, **kwargs)

    @staticmethod
    def prepare_dataset(root, src_lan, tgt_lan, SRC, TGT, load_train_data, max_sequence_length, sentence_count_limit, debug_mode) -> ProcessedData:
        res = ProcessedData()
        dev_data = "valid"
        test_data_list = ['newstest2008', 'newstest2009', 'newstest2010', 'newstest2011', 'newstest2012', 'newstest2013', 'newstest2019',
                          'euelections_dev2019']
        train_data = "train"
        if not load_train_data:
            val, *test = WMT19DeFr.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)), fields=(SRC, TGT), train=None,
                                          validation="valid" if dev_data == "valid" else '{}-defr'.format(dev_data),
                                          test_list=['{}-defr'.format(test_data) for test_data in test_data_list],
                                          sentence_count_limit=sentence_count_limit, root=root)
            train = None
        elif max_sequence_length > 1:
            train, val, *test = WMT19DeFr.splits(
                filter_pred=lambda x: len(vars(x)['src']) <= max_sequence_length and len(
                    vars(x)['trg']) <= max_sequence_length, exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
                fields=(SRC, TGT), train=train_data, validation="valid" if dev_data == "valid" else '{}-defr'.format(dev_data),
                test_list=['{}-defr'.format(test_data) for test_data in test_data_list], sentence_count_limit=sentence_count_limit, root=root)
        else:
            train, val, *test = WMT19DeFr.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)), fields=(SRC, TGT), train=train_data,
                                                 validation="valid" if dev_data == "valid" else '{}-defr'.format(dev_data),
                                                 test_list=['{}-defr'.format(test_data) for test_data in test_data_list],
                                                 sentence_count_limit=sentence_count_limit, root=root)
        if dev_data == "valid":
            res.addresses.val.src = "{}/wmt19_de_fr/valid.{}".format(root, src_lan)
            res.addresses.val.tgt = "{}/wmt19_de_fr/valid.{}".format(root, tgt_lan)
            # TODO fill out this part
            res.addresses.val.src_sgm = None
            res.addresses.val.tgt_sgm = None
        else:
            res.addresses.val.src = "{}/wmt19_de_fr/{}-defr.{}".format(root, dev_data, src_lan)
            res.addresses.val.tgt = "{}/wmt19_de_fr/{}-defr.{}".format(root, dev_data, tgt_lan)
            # TODO fill out this part
            res.addresses.val.src_sgm = None
            res.addresses.val.tgt_sgm = None
        res.addresses.tests.src = ["{}/wmt19_de_fr/{}-defr.{}".format(
            root, test_data, src_lan) for test_data in test_data_list]
        res.addresses.tests.tgt = ["{}/wmt19_de_fr/{}-defr.{}".format(
            root, test_data, tgt_lan) for test_data in test_data_list]
        # TODO fill out this part
        res.addresses.tests.src_sgm = None
        res.addresses.tests.tgt_sgm = None
        res.addresses.train.src = "{}/wmt19_de_fr/{}.{}".format(root, train_data, src_lan)
        res.addresses.train.tgt = "{}/wmt19_de_fr/{}.{}".format(root, train_data, tgt_lan)
        return res


def get_dataset_from_configs(root, dataset_name, src_lan, tgt_lan, SRC, TGT, load_train_data, max_sequence_length=-1,
                             sentence_count_limit=-1, debug_mode=False) -> ProcessedData:
    if dataset_name == "multi30k16":
        return M30k.prepare_dataset(root, src_lan, tgt_lan, SRC, TGT, load_train_data, max_sequence_length, sentence_count_limit, debug_mode)
    elif dataset_name == "iwslt17":
        return IWSLT.prepare_dataset(root, src_lan, tgt_lan, SRC, TGT, load_train_data, max_sequence_length, sentence_count_limit, debug_mode)
    elif dataset_name == "wmt19_de_en":
        return WMT19DeEn.prepare_dataset(root, src_lan, tgt_lan, SRC, TGT, load_train_data, max_sequence_length, sentence_count_limit, debug_mode)
    elif dataset_name == "wmt19_de_fr":
        return WMT19DeFr.prepare_dataset(root, src_lan, tgt_lan, SRC, TGT, load_train_data, max_sequence_length, sentence_count_limit, debug_mode)
    else:
        raise ValueError("A dataset equivalent to the name {} is not implemented!".format(dataset_name))
