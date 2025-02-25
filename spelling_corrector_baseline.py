import nltk
from collections import Counter
import pickle

EPS = 1e-12

class ContextSensitiveSpellCorrector:
    def __init__(self, ngram_model_path, n=3):
        with open(ngram_model_path, "rb") as f:
            ngram_model = pickle.load(f)
        self.ngram_model = ngram_model
        self.vocab = set(word for counts in ngram_model.values() for word in counts)
        self.n = n

    def edits1(self, word):
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]

        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def generate_candidates(self, word, max_distance=2):
        candidates = {word} if word in self.vocab else set()
        valid_edits1 = self.edits1(word) & self.vocab
        if valid_edits1:
            return valid_edits1

        if max_distance > 1:
            word_edits2 = self.edits2(word)
            return word_edits2 & self.vocab
        return candidates

    def get_ngram_probability(self, context, word):
        ngram = tuple(context)
        if ngram not in self.ngram_model:
            return EPS
        if word not in self.ngram_model[ngram]:
            return EPS
        return self.ngram_model.get(ngram, EPS)[word]

    def correct_sentence(self, sentence):
        words = nltk.word_tokenize(sentence.lower())
        corrected_words = []

        for i, word in enumerate(words):
            if word in self.vocab:
                corrected_words.append(word)
                continue

            context = tuple(corrected_words[-(self.n-1):]) if i >= self.n-1 else tuple(corrected_words)
            candidates = self.generate_candidates(word)
            if not candidates:
                corrected_words.append(word)
                continue

            best_candidate = max(candidates, key=lambda w: self.get_ngram_probability(context, w))
            corrected_words.append(best_candidate)
        return ' '.join(corrected_words)


# usage

# ngram_model_path = r"models\trigram_model_shakespeare.pkl"
# spell_corrector = ContextSensitiveSpellCorrector(ngram_model_path, n=3)

# sentence = "to be or not to ve"
# corrected_sentence = spell_corrector.correct_sentence(sentence)
# print("  initial: ", sentence)
# print("corrected: ", corrected_sentence)
