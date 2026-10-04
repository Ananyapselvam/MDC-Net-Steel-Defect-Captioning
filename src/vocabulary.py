import re
from collections import Counter


class Vocabulary:

    def __init__(self, freq_threshold=1):

        # Special tokens
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }

        self.stoi = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }

        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):

        text = str(text).lower()

        # Keep words such as rolled-in-scale together
        text = re.sub(r"[^a-z0-9_-]+", " ", text)

        return text.split()

    def build_vocab(self, sentence_list):

        frequencies = Counter()

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

        idx = 4

        for word, count in frequencies.items():

            if count >= self.freq_threshold:

                self.stoi[word] = idx
                self.itos[idx] = word

                idx += 1

    def numericalize(self, text):

        tokens = self.tokenize(text)

        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokens
        ]

    def decode(self, numericalized_sentence):

        return " ".join(
            self.itos.get(idx, "<UNK>")
            for idx in numericalized_sentence
        )