# -*- coding: utf-8 -*-
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import sys
import os
import unittest
from readers.tokenizers import PreTrainedTokenizer, PyMosesTokenizer, GenericTokenizer
import unidecode


class TestMosesTokenizer(unittest.TestCase):
    """Advanced test cases."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.maxDiff = 1300
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.data/m30k/'))
        if not os.path.exists(data_dir):
            raise ValueError("The WMT validation data is not accessible under {}".format(data_dir))
        cls.moses_exception_sentences = {
            "en":
                [
                    # wrong join of character \'
                    "Two people are holding a large upside-down earth globe, about 4' in diameter, and a child appears to be jumping over Antarctica.",
                    # ... is merged with character before
                    "A man in a blue shirt is holding a sign that says \"Come on now ... what's gayer than tea.\"",
                ],
            "de":
                [
                    # \xa0 is inside the string
                    "Zwei Menschen halten einen großen umgekehrten Globus mit einem Durchmesser von ca. 120 cm und es wirkt so, als würde ein Kind über die Antarktis springen.",
                    # ... is merged with character before
                    "Ein Mann in einem blauen Hemd hält ein Schild, auf dem steht: „Come on now ... what's gayer than tea.“"
                ],
            "fr":
                [
                    # . should be outside of quote but is inside
                    "Beaucoup d'enfants asiatiques ont fait le train sous un panneau \"Viet Nam\".",
                    "Un homme en polo bleue tient une pancarte qui dit : \"Allez quoi... qu'est-ce qu'il y a de plus gay que le thé\".",
                    "Deux personnes, l'une vêtue comme une religieuse et l'autre en t-shirt \"roger smith\", engagées dans une course à pied, dépassant les spectateurs dans une zone boisée.",
                    # ` and ' characters are different (un-normalized source)
                    "Une femme jette un coup d’œil dans un télescope dans les bois."
                ]
        }
        cls.pre_trained_exception_sentences = {  # UNK cases
            "en":
                [],
            "de":
                [],
            "fr":
                []
        }
        cls.valid_sentences = {}
        for lang in ["en", "de", "fr"]:
            with open(os.path.join(data_dir, 'val.{}'.format(lang)), "r", encoding="utf-8") as f:
                cls.valid_sentences[lang] = [line.strip() for line in f]
            print("Total valid sentences for {} = {}".format(lang, len(cls.valid_sentences[lang])))
            print("Total special sentences for {} = {}".format(lang, len(cls.moses_exception_sentences[lang])))

    def test_generic_tokenizer(self):
        s = GenericTokenizer()
        for lang in self.valid_sentences:
            for sent in self.valid_sentences[lang]:
                sent = sent.strip().replace(u"\xa0", " ")
                recovered = s.detokenize(s.tokenize(sent))
                if "  " not in sent:
                    self.assertEqual(sent, recovered)

    def test_moses_tokenizer(self):
        for lang in self.valid_sentences:
            lowercase = True
            s = PyMosesTokenizer(lang, lowercase)
            for sent in self.valid_sentences[lang]:
                if sent not in self.moses_exception_sentences[lang]:
                    tokens = s.tokenize(sent)
                    recovered = s.detokenize(tokens)
                    expected = sent.lower() if lowercase else sent
                    expected = " ".join([x for x in expected.split() if x])
                    if lang == "de":
                        expected = expected.replace("„", "\"").replace("“", "\"").replace("–", "-")
                    self.assertEqual(expected, recovered)

    def test_pre_trained_tokenizer(self):
        for lang in self.valid_sentences:
            lowercase = lang == "fr"  # lowercasing makes much more [UNK]s
            s = PreTrainedTokenizer(lang, lowercase=lowercase)
            for sent in self.valid_sentences[lang]:
                if sent not in self.moses_exception_sentences[lang] and sent not in self.pre_trained_exception_sentences[lang]:
                    tokens = s.tokenize(sent)
                    recovered = s.detokenize(tokens)
                    expected = sent.lower() if lowercase else sent
                    expected = " ".join([x for x in expected.split() if x])
                    if lang == "de":
                        expected = expected.replace("„", "\"").replace("“", "\"").replace("–", "-")
                    if unidecode.unidecode(sent) == sent:
                        self.assertEqual(expected, recovered)
                    else:
                        self.assertEqual(unidecode.unidecode(expected), unidecode.unidecode(recovered))


if __name__ == '__main__':
    unittest.main()
