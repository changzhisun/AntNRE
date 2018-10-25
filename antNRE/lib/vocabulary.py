#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/09/17 17:22:49

@author: Changzhi Sun
"""
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"

def namespace_match(pattern: str, namespace: str):
    if pattern[0] == "*" and namespace.endswith(pattern[1:]):
        return True
    elif pattern == namespace:
        return True
    return False

class _NamespaceDependentDefaultDict(defaultdict):
    """
    This is a `defaultdict
    <https://docs.python.org/2/library/collections.html#collections.defaultdict>`_ where the
    default value is dependent on the key that is passed.

    We use "namespaces" in the :class:`Vocabulary` object to keep track of several different
    mappings from strings to integers, so that we have a consistent API for mapping words, tags,
    labels, characters, or whatever else you want, into integers.  The issue is that some of those
    namespaces (words and characters) should have integers reserved for padding and
    out-of-vocabulary tokens, while others (labels and tags) shouldn't.  This class allows you to
    specify filters on the namespace (the key used in the ``defaultdict``), and use different
    default values depending on whether the namespace passes the filter.

    To do filtering, we take a set of ``non_padded_namespaces``.  This is a set of strings
    that are either matched exactly against the keys, or treated as suffixes, if the
    string starts with ``*``.  In other words, if ``*tags`` is in ``non_padded_namespaces`` then
    ``passage_tags``, ``question_tags``, etc. (anything that ends with ``tags``) will have the
    ``non_padded`` default value.

    Parameters
    ----------
    non_padded_namespaces : ``Iterable[str]``
        A set / list / tuple of strings describing which namespaces are not padded.  If a namespace
        (key) is missing from this dictionary, we will use :func:`namespace_match` to see whether
        the namespace should be padded.  If the given namespace matches any of the strings in this
        list, we will use ``non_padded_function`` to initialize the value for that namespace, and
        we will use ``padded_function`` otherwise.
    padded_function : ``Callable[[], Any]``
        A zero-argument function to call to initialize a value for a namespace that `should` be
        padded.
    non_padded_function : ``Callable[[], Any]``
        A zero-argument function to call to initialize a value for a namespace that should `not` be
        padded.
    """
    def __init__(self,
                 non_padded_namespaces: Iterable[str],
                 padded_function: Callable[[], Any],
                 non_padded_function: Callable[[], Any]) -> None:
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super(_NamespaceDependentDefaultDict, self).__init__()

    def __missing__(self, key: str):
        if any(namespace_match(pattern, key) for pattern in self._non_padded_namespaces):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value

    def add_non_padded_namespaces(self, non_padded_namespaces: Set[str]):
        # add non_padded_namespaces which weren't already present
        self._non_padded_namespaces.update(non_padded_namespaces)

class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super(_TokenToIndexDefaultDict, self).__init__(non_padded_namespaces,
                                                       lambda: {padding_token: 0, oov_token: 1},
                                                       lambda: {})

class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super(_IndexToTokenDefaultDict, self).__init__(non_padded_namespaces,
                                                       lambda: {0: padding_token, 1: oov_token},
                                                       lambda: {})
class Vocabulary:
    """
    A Vocabulary maps strings to integers, allowing for strings to be mapped to an
    out-of-vocabulary token.

    Vocabularies are fit to a particular dataset, which we use to decide which tokens are
    in-vocabulary.

    Vocabularies also allow for several different namespaces, so you can have separate indices for
    'a' as a word, and 'a' as a character, for instance, and so we can use this object to also map
    tag and label strings to indices, for a unified :class:`~.fields.field.Field` API.  Most of the
    methods on this class allow you to pass in a namespace; by default we use the 'tokens'
    namespace, and you can omit the namespace argument everywhere and just use the default.

    Parameters
    ----------
    counter : ``Dict[str, Dict[str, int]]``, optional (default=``None``)
        A collection of counts from which to initialize this vocabulary.  We will examine the
        counts and, together with the other parameters to this class, use them to decide which
        words are in-vocabulary.  If this is ``None``, we just won't initialize the vocabulary with
        anything.
    min_count : ``Dict[str, int]``, optional (default=None)
        When initializing the vocab from a counter, you can specify a minimum count, and every
        token with a count less than this will not be added to the dictionary.  These minimum
        counts are `namespace-specific`, so you can specify different minimums for labels versus
        words tokens, for example.  If a namespace does not have a key in the given dictionary, we
        will add all seen tokens to that namespace.
    max_vocab_size : ``Union[int, Dict[str, int]]``, optional (default=``None``)
        If you want to cap the number of tokens in your vocabulary, you can do so with this
        parameter.  If you specify a single integer, every namespace will have its vocabulary fixed
        to be no larger than this.  If you specify a dictionary, then each namespace in the
        ``counter`` can have a separate maximum vocabulary size.  Any missing key will have a value
        of ``None``, which means no cap on the vocabulary size.
    non_padded_namespaces : ``Iterable[str]``, optional
        By default, we assume you are mapping word / character tokens to integers, and so you want
        to reserve word indices for padding and out-of-vocabulary tokens.  However, if you are
        mapping NER or SRL tags, or class labels, to integers, you probably do not want to reserve
        indices for padding and out-of-vocabulary tokens.  Use this field to specify which
        namespaces should `not` have padding and OOV tokens added.

        The format of each element of this is either a string, which must match field names
        exactly,  or ``*`` followed by a string, which we match as a suffix against field names.

        We try to make the default here reasonable, so that you don't have to think about this.
        The default is ``("*tags", "*labels")``, so as long as your namespace ends in "tags" or
        "labels" (which is true by default for all tag and label fields in this code), you don't
        have to specify anything here.
    pretrained_files : ``Dict[str, str]``, optional
        If provided, this map specifies the path to optional pretrained embedding files for each
        namespace. This can be used to either restrict the vocabulary to only words which appear
        in this file, or to ensure that any words in this file are included in the vocabulary
        regardless of their count, depending on the value of ``only_include_pretrained_words``.
        Words which appear in the pretrained embedding file but not in the data are NOT included
        in the Vocabulary.
    only_include_pretrained_words : ``bool``, optional (default=False)
        This defines the stategy for using any pretrained embedding files which may have been
        specified in ``pretrained_files``. If False, an inclusive stategy is used: and words
        which are in the ``counter`` and in the pretrained file are added to the ``Vocabulary``,
        regardless of whether their count exceeds ``min_count`` or not. If True, we use an
        exclusive strategy: words are only included in the Vocabulary if they are in the pretrained
        embedding file (their count must still be at least ``min_count``).
    tokens_to_add : ``Dict[str, List[str]]``, optional (default=None)
        If given, this is a list of tokens to add to the vocabulary, keyed by the namespace to add
        the tokens to.  This is a way to be sure that certain items appear in your vocabulary,
        regardless of any other vocabulary computation.
    """
    def __init__(self,
                 counter: Dict[str, Dict[str, int]] = None,
                 min_count: Dict[str, int] = None,
                 max_vocab_size: Union[int, Dict[str, int]] = None,
                 non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
                 pretrained_files: Optional[Dict[str, str]] = None,
                 only_include_pretrained_words: bool = False,
                 tokens_to_add: Dict[str, List[str]] = None) -> None:
        self._padding_token = DEFAULT_PADDING_TOKEN
        self._oov_token = DEFAULT_OOV_TOKEN
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._token_to_index = _TokenToIndexDefaultDict(self._non_padded_namespaces,
                                                        self._padding_token,
                                                        self._oov_token)
        self._index_to_token = _IndexToTokenDefaultDict(self._non_padded_namespaces,
                                                        self._padding_token,
                                                        self._oov_token)
        # Made an empty vocabulary, now extend it.
        self.extend(counter,
                    min_count,
                    max_vocab_size,
                    non_padded_namespaces,
                    pretrained_files,
                    only_include_pretrained_words,
                    tokens_to_add)

    def extend(self,
               counter: Dict[str, Dict[str, int]] = None,
               min_count: Dict[str, int] = None,
               max_vocab_size: Union[int, Dict[str, int]] = None,
               non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
               pretrained_files: Optional[Dict[str, str]] = None,
               only_include_pretrained_words: bool = False,
               tokens_to_add: Dict[str, List[str]] = None) -> None:
        """
        This method can be used for extending already generated vocabulary.
        It takes same parameters as Vocabulary initializer. The token2index
        and indextotoken mappings of calling vocabulary will be retained.
        It is an inplace operation so None will be returned.
        """
        if not isinstance(max_vocab_size, dict):
            int_max_vocab_size = max_vocab_size
            max_vocab_size = defaultdict(lambda: int_max_vocab_size)  # type: ignore
        min_count = min_count or {}
        pretrained_files = pretrained_files or {}
        non_padded_namespaces = set(non_padded_namespaces)

        if counter is not None:
            # Make sure vocabulary extension is safe.
            for namespace in counter:
                if namespace in self.get_all_namespaces():
                    # if new namespace was already present
                    # Either both should be padded or none should be.
                    original_padded = not any(namespace_match(pattern, namespace)
                                              for pattern in self._non_padded_namespaces)
                    extension_padded = not any(namespace_match(pattern, namespace)
                                               for pattern in non_padded_namespaces)
                    if original_padded != extension_padded:
                        raise ConfigurationError("Common namespace {} has conflicting ".format(namespace)+
                                                 "setting of padded = True/False. "+
                                                 "Hence extension cannot be done.")
            # Add new non-padded namespaces for extension
            self.add_non_padded_namespaces(non_padded_namespaces)

            for namespace in counter:
                if namespace in pretrained_files:
                    pretrained_list = _read_pretrained_tokens(pretrained_files[namespace])
                else:
                    pretrained_list = None
                token_counts = list(counter[namespace].items())
                token_counts.sort(key=lambda x: x[1], reverse=True)
                max_vocab = max_vocab_size[namespace]
                if max_vocab:
                    token_counts = token_counts[:max_vocab]
                for token, count in token_counts:
                    if pretrained_list is not None:
                        if only_include_pretrained_words:
                            if token in pretrained_list and count >= min_count.get(namespace, 1):
                                self.add_token_to_namespace(token, namespace)
                        elif token in pretrained_list or count >= min_count.get(namespace, 1):
                            self.add_token_to_namespace(token, namespace)
                    elif count >= min_count.get(namespace, 1):
                        self.add_token_to_namespace(token, namespace)

        if tokens_to_add:
            for namespace, tokens in tokens_to_add.items():
                for token in tokens:
                    self.add_token_to_namespace(token, namespace)

    def add_non_padded_namespaces(self, non_padded_namespaces: Set[str]):
        self._token_to_index.add_non_padded_namespaces(non_padded_namespaces)
        self._index_to_token.add_non_padded_namespaces(non_padded_namespaces)
        self._non_padded_namespaces.update(non_padded_namespaces)

    def get_all_namespaces(self) -> Set[str]:
        return set(self._token_to_index.keys())

    def get_non_padded_namespaces(self) -> Set[str]:
        return self._non_padded_namespaces

    def get_token_to_index(self) -> Dict[str, Dict[str, int]]:
        return self._token_to_index

    def get_index_to_token(self) -> Dict[str, Dict[int, str]]:
        return self._index_to_token

    def is_padded(self, namespace: str) -> bool:
        """
        Returns whether or not there are padding and OOV tokens added to the given namepsace.
        """
        return self._index_to_token[namespace][0] == self._padding_token

    def add_token_to_namespace(self, token: str, namespace: str = 'tokens') -> int:
        """
        Adds ``token`` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if not isinstance(token, str):
            raise ValueError("Vocabulary tokens must be strings, or saving and loading will break."
                             "  Got %s (with type %s)" % (repr(token), type(token)))
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            return index
        else:
            return self._token_to_index[namespace][token]

    def get_index_to_token_vocabulary(self, namespace: str = 'tokens') -> Dict[int, str]:
        return self._index_to_token[namespace]

    def get_token_to_index_vocabulary(self, namespace: str = 'tokens') -> Dict[str, int]:
        return self._token_to_index[namespace]

    def get_token_index(self, token: str, namespace: str = 'tokens') -> int:
        if token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][token]
        else:
            try:
                return self._token_to_index[namespace][self._oov_token]
            except KeyError:
                logger.error('Namespace: %s', namespace)
                logger.error('Token: %s', token)
                raise

    def get_token_from_index(self, index: int, namespace: str = 'tokens') -> str:
        return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str = 'tokens') -> int:
        return len(self._token_to_index[namespace])

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self) -> str:
        base_string = f"Vocabulary with namespaces:\n"
        non_padded_namespaces = f"\tNon Padded Namespaces: {self._non_padded_namespaces}\n"
        namespaces = [f"\tNamespace: {name}, Size: {self.get_vocab_size(name)} \n"
                      for name in self._index_to_token]
        return " ".join([base_string, non_padded_namespaces] + namespaces)
