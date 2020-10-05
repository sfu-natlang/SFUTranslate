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
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.data/iwslt/de-en/'))
        if not os.path.exists(data_dir):
            raise ValueError("The WMT validation data is not accessible under {}".format(data_dir))
        cls.moses_exception_sentences = {
            "en":
                [
                    "Video: ♪♫ Frosty the coal man is a jolly, happy soul.",  # music signs are separated in moses
                    "Man #2: New investments to create high-paying jobs.",  # # and 2 are separated

                    # ... is joined with the last word in sentence

                    "Sunlight and CO2 is one method ...",
                    "We truly, you know,  have modest goals  of replacing the whole petrol-chemical industry --     Yeah. If you can't do that at TED, where can you? --    become a major source of energy ...",

                    # ending punctuation and is separate from " which is the last character in sentence
                    "I mean, don't you know anything?\" And I said, \"No.\"",  # not like others: i mean, don't you know anything? "and i said," no. "
                    "B has these benefits, and these risks. What do you want to do?\"",
                    # "And you say, \"Doc, what should I do?\"" ====> Not this one
                    # "And you say, \"If you were me, Doc, what would you do?\"" ====> Not this one
                    "You want boot cut, tapered, blah blah.\" On and on he went.",
                 ],
            "de":
                [
                    "♪♫ Frosty der Kohlenmann ist ein vergnügter, fröhlicher Bursche.",
                    "Als Informatiker - inspiriert von der Art unserer Interaktion mit realen Objekten - zusammen mit meinem Fachberater Patti und meinem Kollegen Jeevan Kalanithi, begann ich mich zu wundern: Was wäre, wenn wir Computer nutzen würden, die anstelle eines Maus-Zeigers, der sich wie eine digitale Fingerspitze auf einer flachen Arbeitsfläche bewegt... ... Was wäre, wenn wir mit beiden Händen eingreifen könnten und Daten physisch ergreifen",
                    "...seine eigene Geschichte erfinden.",
                    "Denn...  ...am Ende ist es wie folgt. Vor Jahrhunderten haben sich in den Wüsten Nordafrikas Menschen zu Mondscheinfesten mit heiligen Tänzen und Musik versammelt, die stundenlang abgehalten wurden – bis zur Morgendämmerung.",
                    "Möchten Sie unten weit geschnitten, kegelig, blah blah blah ...\" und so weiter erzählte er.",
                    "Die Griechen nannten diese göttlichen Diener-Geister der Kreativität „Dämonen“.",
                    "Die Römer hatten die gleiche Idee, nannten diese Art von körperlosem kreativem Geist ein „Genie“.",
                    "Sie wusste, dass sie in einem solchen Moment nur eines tun konnte. Und das war – in ihren Worten – „rennen wie der Teufel“.",
                    "Bei anderen Malen war sie nicht schnell genug. Sie rannte und rannte und rannte, aber sie erreichte das Haus nicht und das Gedicht rollte durch sie hindurch und sie verpasste es. Sie sagte es würde weiter über Land ziehen und – wie sie sagte – „nach einem anderen Dichter suchen“.",
                    "Interessante historische Fußnote: als die Mohren Südspanien eroberten brachten sie diesen Brauch mit. Die Aussprache änderte sich im Laufe der Jahrhunderte von „Allah, Allah, Allah“ zu „Olé, olé, olé“, das man immer noch bei Stierkämpfen und Flamenco-Tänzen hört.",
                    "Und ich weiß, dass einige von Ihnen sagen: \"Ist es nicht besser so?\" Wäre die Welt nicht ein besserer Ort, wenn wir alle nur eine Sprache sprechen würden?\" Und ich sage:\"Gut, lass diese Sprache Yoruba sein. Lass sie Kantonesisch sein.",
                    "Ich meine, weißt du überhaupt nichts?\" Und ich sagte: \"Nein.\"",
                    "Das Ergebnis nennen wir \"Patientenautonomie\". Das klingt wie eine gute Sache, aber es verschiebt die Last der Verantwortung für das Treffen von Entscheidungen von jemandem der etwas weiß, nämlich vom Arzt, zu jemandem der nichts weiß und höchstwahrscheinlich krank ist, und daher nicht in der besten Verfassung, um zu entscheiden -- nämlich der Patient.",
                    "Die einzige wahre Wahl war \"wen\", nicht wann, und nicht was Sie danach taten.",
                    "Ich unterrichte wunderbar intelligente Studenten, und ich gebe ihnen 20 % weniger Arbeit als früher.",
                    "Die Antwort ist \"Ja\".",
                    "Durch die nicht Teilnahme geben Sie bis zu 5000 US$ pro Jahr auf von dem Arbeitsgeber, der glücklich gewesen wäre Ihren Beitrag zu vermehren."
                ]
        }
        cls.pre_trained_exception_sentences = {  # UNK cases
            "en":
                [],
            "de":
                []
        }
        cls.valid_sentences = {}
        for lang in ["en", "de"]:
            with open(os.path.join(data_dir, 'IWSLT17.TED.dev2010.de-en.{}'.format(lang)), "r", encoding="utf-8") as f:
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
                    if lang == "de":
                        expected = expected.replace("„", "\"").replace("“", "\"").replace("–", "-")
                    self.assertEqual(" ".join([x for x in expected.split() if x]), recovered)

    def test_pre_trained_tokenizer(self):
        for lang in self.valid_sentences:
            lowercase = False  # lowercasing makes much more [UNK]s
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

    def test_french_pretrained_tokenizer(self):
        lang = "fr"
        lowercase = True
        s = PreTrainedTokenizer(lang, lowercase=lowercase)
        for sent in ["Monsieur le Président, ce qui s'est déroulé ces derniers mois dans les îles Fidji semblait inspiré d'un feuilleton de l'après-midi."]:
            tokens = s.tokenize(sent)
            recovered = s.detokenize(tokens)
            if unidecode.unidecode(sent) == sent:
                self.assertEqual(sent.lower() if lowercase else sent, recovered)
            else:
                self.assertEqual(unidecode.unidecode(sent.lower()) if lowercase else unidecode.unidecode(sent), unidecode.unidecode(recovered))


if __name__ == '__main__':
    unittest.main()
