"""
The modified implementations of torchtext.datasets classes containing updated dataset urls as well as extension checking and a unified splits function
"""
import os
import io
import glob
import codecs
import xml.etree.ElementTree as ET

from readers.datasets.generic import TranslationDataset


class M30k(TranslationDataset):
    """The small-dataset in WMT 2017 multimodal task"""

    urls = ['http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_training.tar.gz',
            'http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_validation.tar.gz',
            'http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_test2016.tar.gz']
    name = 'm30k'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train', validation='val', test_list=('test2016',), **kwargs):
        if exts[0][1:] not in ['fr', 'en', 'de'] or exts[1][1:] not in ['fr', 'en', 'de']:
            raise ValueError("This data set only contains data translated to/from English, French, or German")
        return super(M30k, cls).splits(exts, fields, None, root, train, validation, test_list, **kwargs)


class IWSLT(TranslationDataset):
    """
    Do-over of the original Torchtext IWSLT Library.
    This one does not apply filter_pred designed for length limitation to the test and dev datasets.
    The IWSLT 2017 TED talk translation task - https://wit3.fbk.eu/mt.php?release=2017-01-trnted
    """

    # base_url = 'https://wit3.fbk.eu/archive/2016-01//texts/{}/{}/{}.tgz'
    base_url = 'https://wit3.fbk.eu/archive/2017-01-trnted/texts/{}/{}/{}.tgz'
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
        cls.urls = [cls.base_url.format(exts[0][1:], exts[1][1:], cls.dirname)]
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


class WMT19DeEn(TranslationDataset):
    """The WMT 2019 English-German dataset. The download and data preparation script is enclosed in the dataset folder"""

    urls = [('https://drive.google.com/uc?export=download&'
             'id=1miCyP1Vdoi6QsGoKSWhNgxbPVmSSs3Rt', 'wmt19_en_de_raw.zip')]
    name = 'wmt19_en_de'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data', train='train', validation='valid',
               test_list=('newstest2014-ende', 'newstest2015-ende', 'newstest2016-ende',
                          'newstest2017-ende', 'newstest2018-ende', 'newstest2019-ende'), **kwargs):
        if exts[0][1:] not in ['en', 'de'] or exts[1][1:] not in ['en', 'de']:
            raise ValueError("This data set only contains data translated from German to English or reverse")
        return super(WMT19DeEn, cls).splits(exts, fields, None, root, train, validation, test_list, **kwargs)


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
