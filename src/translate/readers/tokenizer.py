"""
The wrapper over the spaCy's Tokenizer for `English`,`German`,`Spanish`,`Portuguese`,`French`,`Italian`, and `Dutch`.
 Based on the library's documentation website, the tokenization algorithm can be summarized as follows:
    1. Iterate over space-separated substrings
    2. Check whether we have an explicitly defined rule for this substring. If we do, use it.
    3. Otherwise, try to consume a prefix.
    4. If we consumed a prefix, go back to the beginning of the loop, so that special-cases always get priority.
    5. If we didn't consume a prefix, try to consume a suffix.
    6. If we can't consume a prefix or suffix, look for "infixes" — stuff like hyphens etc.
    7. Once we can't consume any more of the string, handle it as a single token.
For more info regarding the tokenizer please see the "Tokenization" part of https://spacy.io/usage/linguistic-features
A valid use case of the tokenizer wrapper class could be:
    SpaCyTokenizer().tokenize("This is a test", LanguageIdentifier.en)
"""
from typing import List

import spacy

from translate.readers.constants import LanguageIdentifier as LId

__author__ = "Hassan S. Shavarani"


class SpaCyTokenizer:
    def __init__(self):
        """
        The tokenizer performs lazy instantiation of the models. You don't need multiple instances of this class for
         tokenization of sentences from different languages.
        """
        self._models = {}
        self._supported_languages = [LId.en, LId.de, LId.es, LId.pt, LId.fr, LId.it, LId.nl]

    def tokenize(self, text: str, lang_identifier: LId, lower_case: bool = False) -> List[str]:
        """
        :param text: the string to be tokenized
        :param lang_identifier: one of the langugage values defined in `translate.readers.constants.LanguageIdentifier`
        :param lower_case: the flag indicating whether the resulting tokens need to be lower-cased or not.
        :return: the list of tokenized strings
        """
        if lang_identifier not in self._supported_languages:
            raise ValueError("SpaCyTokenizer cannot tokenize utterances in \"{}\"".format(lang_identifier.name))
        if lang_identifier not in self._models:
            try:
                self._models[lang_identifier] = spacy.load(lang_identifier.name)
            except OSError:
                raise EnvironmentError("The spaCy resources for \"{0}\" might not be installed correctly, please try "
                                       "running the following command in your comman-line before running this project\n"
                                       "python -m spacy download {0}".format(lang_identifier.name))
        tokenized_document = self._models[lang_identifier].tokenizer(text)
        if lower_case:
            return [token.text.lower() for token in tokenized_document]
        else:
            return [token.text for token in tokenized_document]
