from typing import List, Union, Optional, Tuple
import numpy as np
from tqdm.notebook import tqdm
import pickle


class PrefixTreeNode:
    def __init__(self, word: str = ""):
        self.children: dict[str, PrefixTreeNode] = {}
        self.word = word
        self.is_end_of_word = False

    def get_all_children(self):
        all_children = []
        if self.is_end_of_word:
            all_children.append(self)
        for char in self.children:
            child = self.children[char]
            if child.is_end_of_word:
                all_children.append(child)
            all_children.extend(child.get_all_children())
        return all_children


class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: список всех уникальных токенов в корпусе
        """
        self.root = PrefixTreeNode()

        for word in vocabulary:
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = PrefixTreeNode(node.word + char)
                node = node.children[char]
            node.is_end_of_word = True

    def search_prefix(self, prefix) -> List[str]:
        """
        Возвращает все слова, начинающиеся на prefix
        prefix: str – префикс слова
        """

        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        children = node.get_all_children()
        children_words = [child.word for child in children]

        return list(set(children_words))


class WordCompletor:
    def __init__(self, corpus, border=0.0):
        self.word_counts = {}
        self.n_corpus_words = 0
        for text in tqdm(corpus):
            if isinstance(text, str):
                text = text.split()
            self.n_corpus_words += len(text)
            for word in text:
                if word not in self.word_counts:
                    self.word_counts[word] = 0
                self.word_counts[word] += 1
        vocab = list(self.word_counts.keys())
        if border < 1.0:
            abs_border = np.quantile(list(self.word_counts.values()), border)
        else:
            abs_border = round(border)
        for word in vocab:
            if self.word_counts[word] < abs_border:
                self.word_counts.pop(word)
        vocab = list(self.word_counts.keys())
        self.prefix_tree = PrefixTree(vocab)
        print("vocab size:", len(vocab))
        print("WordCompletor.__init__ finished")

    def get_words_and_probs(
        self, prefix: str, max_words: Union[int, None] = None
    ) -> Tuple[List[str], List[float]]:
        words = self.prefix_tree.search_prefix(prefix)
        probs = [self.word_counts[word] / self.n_corpus_words for word in words]

        probs_argsort = np.asarray(probs).argsort()[::-1]
        if max_words is not None:
            probs_argsort = probs_argsort[:max_words]

        words = (np.asarray(words, dtype="str")[probs_argsort]).tolist()
        probs = (np.asarray(probs)[probs_argsort]).tolist()

        return words, probs


class NGramLanguageModel:
    def __init__(self, corpus, n):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}

        for text in tqdm(corpus):
            if isinstance(text, str):
                text = text.split()
            for i in range(len(text) - n):
                context = tuple(text[i : i + n])
                next_word = text[i + n]
                context_extended = context + (next_word,)

                if context_extended not in self.ngram_counts:
                    self.ngram_counts[context_extended] = 0
                self.ngram_counts[context_extended] += 1

                if context not in self.context_counts:
                    self.context_counts[context] = 0
                self.context_counts[context] += 1

    def get_next_words_and_probs(
        self, prefix: list, max_words: Union[int, None] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Возвращает список слов, которые могут идти после prefix,
        а так же список вероятностей этих слов.
        Возвращает не более max_words слов.
        """

        next_words, probs = [], []

        if len(prefix) < self.n:
            return next_words, probs

        prefix = tuple(prefix[-self.n :])

        if prefix not in self.context_counts:
            return next_words, probs

        for context in self.ngram_counts:
            if context[: self.n] == prefix:
                next_word = context[self.n]
                next_words.append(next_word)
                probs.append(self.ngram_counts[context] / self.context_counts[prefix])

        probs_argsort = np.asarray(probs).argsort()[::-1]
        if max_words is not None:
            probs_argsort = probs_argsort[:max_words]

        next_words = (np.asarray(next_words, dtype="str")[probs_argsort]).tolist()
        probs = (np.asarray(probs)[probs_argsort]).tolist()

        return next_words, probs

    def get_next_seqs_and_probs(
        self,
        prefix: list,
        seq_len: int = 1,
        max_seqs: Union[int, None] = None,
        max_best_words: Union[int, None] = None,
    ) -> Tuple[List[List[str]], List[float]]:
        """
        Возвращает список продолжений для prefix, продолжение состоит из seq_len слов.
        Возвращает не более max_seqs вариантов продолжений.
        """
        next_seqs, probs = [], []
        next_words, word_probs = self.get_next_words_and_probs(prefix, max_best_words)

        if seq_len == 1:
            return [[word] for word in next_words], word_probs

        for word, word_prob in zip(next_words, word_probs):
            local_seqs, local_probs = self.get_next_seqs_and_probs(
                prefix + [word], seq_len - 1, max_seqs, max_best_words
            )
            local_seqs = [[word] + local_seq for local_seq in local_seqs]
            local_probs = [word_prob * local_prob for local_prob in local_probs]

            next_seqs.extend(local_seqs)
            probs.extend(local_probs)

        probs_argsort = np.asarray(probs).argsort()[::-1]
        if max_seqs is not None:
            probs_argsort = probs_argsort[:max_seqs]

        next_seqs = (np.asarray(next_seqs, dtype="str")[probs_argsort]).tolist()
        probs = (np.asarray(probs)[probs_argsort]).tolist()

        return next_seqs, probs


class TextSuggestion:
    def __init__(self, word_completor: WordCompletor, n_gram_model: NGramLanguageModel):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model

    def suggest_text(
        self,
        text: Union[str, list],
        n_words: int = 3,
        n_texts: int = 1,
        max_complete_words: Union[int, None] = 1,
        n_best_words: int = 1,
        return_probs: bool = False,
    ) -> (List[List[str]], Optional[List[float]]):  # type: ignore
        """
        Возвращает возможные варианты продолжения текста (по умолчанию только один)

        text: строка или список слов – написанный пользователем текст
        n_words: число слов, которые дописывает n-граммная модель
        n_texts: число возвращаемых продолжений
        max_complete_words: число вариантов дополнения последнего слова
        n_best_words: число вариантов, которые n-граммная модель предлагает для каждого слова

        return: list[list[srt]] – список из n_texts списков слов, по 1 + n_words слов в каждом
        Первое слово – это то, которое WordCompletor дополнил до целого.
        """

        if isinstance(text, str):
            text = text.split()

        # дописываем последнее слово
        complete_words, complete_probs = self.word_completor.get_words_and_probs(
            text[-1], max_complete_words
        )

        if not complete_words:
            return [[text[-1]] for _ in range(n_texts)]

        # для каждого варианта дополнения слова напишем продолжение
        n_sugg_per_completed = int(max(3, np.ceil(2 * n_texts / len(complete_words))))
        suggestions = []
        suggestions_probs = []

        for ind_completed, completed_word in enumerate(complete_words):
            history = text[:-1] + [completed_word]
            local_sugg, local_sugg_probs = self.n_gram_model.get_next_seqs_and_probs(
                history, n_words, n_sugg_per_completed, n_best_words
            )
            local_sugg = [[completed_word] + seq for seq in local_sugg]
            local_sugg_probs = [
                local_prob * complete_probs[ind_completed]
                for local_prob in local_sugg_probs
            ]

            suggestions.extend(local_sugg)
            suggestions_probs.extend(local_sugg_probs)

        probs_argsort = np.asarray(suggestions_probs).argsort()[::-1]
        probs_argsort = probs_argsort[:n_texts]

        suggestions = (np.asarray(suggestions, dtype="str")[probs_argsort]).tolist()
        suggestions_probs = (np.asarray(suggestions_probs)[probs_argsort]).tolist()

        if return_probs:
            return suggestions, suggestions_probs
        return suggestions
