from abc import ABC, abstractmethod
import jellyfish
import nltk
import pickle


EPS = 1e-12

class SpellCorrectorInterface(ABC):
    @abstractmethod
    def edits1(self, word: str) -> set:
        pass

    @abstractmethod
    def edits2(self, word: str) -> set:
        pass

    @abstractmethod
    def generate_candidates(self, word: str) -> set:
        pass

    @abstractmethod
    def get_ngram_probability(self, context: tuple, word: str) -> float:
        pass

    @abstractmethod
    def correct_sentence(self, sentence: str) -> str:
        pass


class SpellCorrector(SpellCorrectorInterface):
    def __init__(self, ngram_model_path, n=3):
        with open(ngram_model_path, "rb") as f:
            ngram_model = pickle.load(f)
        self.ngram_model = ngram_model
        self.vocab = set(word for counts in ngram_model.values() for word in counts)
        self.n = n

    def get_vocab_size(self):
        return len(self.vocab)

    def edits1(self, word: str) -> set:
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]

        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word: str) -> set:
        return {e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)}

    def generate_edit_candidates(self, word: str) -> set:
        valid_edits1 = self.edits1(word) & self.vocab
        if valid_edits1:
            return valid_edits1
        word_edits2 = self.edits2(word)
        return word_edits2 & self.vocab
    
    def generate_candidates(self, word: str) -> set:
        return self.generate_edit_candidates(word)

    def get_ngram_probability(self, context: tuple, word: str) -> float:
        ngram = tuple(context)
        if ngram not in self.ngram_model:
            return EPS
        if word not in self.ngram_model[ngram]:
            return EPS
        return self.ngram_model.get(ngram, EPS)[word]

    def correct_sentence(self, sentence: str) -> str:
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


class SpellCorrectorImproved(SpellCorrectorInterface):
    def __init__(self, ngram_model_path, n=3):
        with open(ngram_model_path, "rb") as f:
            ngram_model = pickle.load(f)
        self.ngram_model = ngram_model
        self.vocab = set(word for counts in ngram_model.values() for word in counts)
        self.n = n
        self.keyboard_neighbors = self._init_keyboard_neighbors()

    def _init_keyboard_neighbors(self):
        return {
            'q': ['w', 'a', 's'],
            'w': ['q', 'e', 'a', 's', 'd'],
            'e': ['w', 'r', 's', 'd', 'f'],
            'r': ['e', 't', 'd', 'f', 'g'],
            't': ['r', 'y', 'f', 'g', 'h'],
            'y': ['t', 'u', 'g', 'h', 'j'],
            'u': ['y', 'i', 'h', 'j', 'k'],
            'i': ['u', 'o', 'j', 'k', 'l'],
            'o': ['i', 'p', 'k', 'l'],
            'p': ['o', 'l'],
            'a': ['q', 'w', 's', 'z', 'x'],
            's': ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c'],
            'd': ['w', 'e', 'r', 's', 'f', 'x', 'c', 'v'],
            'f': ['e', 'r', 't', 'd', 'g', 'c', 'v', 'b'],
            'g': ['r', 't', 'y', 'f', 'h', 'v', 'b', 'n'],
            'h': ['t', 'y', 'u', 'g', 'j', 'b', 'n', 'm'],
            'j': ['y', 'u', 'i', 'h', 'k', 'n', 'm'],
            'k': ['u', 'i', 'o', 'j', 'l', 'm'],
            'l': ['i', 'o', 'p', 'k'],
            'z': ['a', 's', 'x'],
            'x': ['a', 's', 'd', 'z', 'c'],
            'c': ['s', 'd', 'f', 'x', 'v'],
            'v': ['d', 'f', 'g', 'c', 'b'],
            'b': ['f', 'g', 'h', 'v', 'n'],
            'n': ['g', 'h', 'j', 'b', 'm'],
            'm': ['h', 'j', 'k', 'n']
        }

    def edits1_keyboard(self, word: str) -> set:
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        
        replaces = []
        for L, R in splits:
            if R:
                for c in self.keyboard_neighbors.get(R[0], []):
                    replaces.append(L + c + R[1:])
        inserts = [L + c + R for L, R in splits for c in letters]

        return set(deletes + transposes + replaces + inserts)

    def edits1(self, word: str) -> set:
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]

        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word: str) -> set:
        return {e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)}

    def generate_edit_candidates(self, word: str) -> set:
        valid_edits1 = self.edits1(word) & self.vocab
        if valid_edits1:
            return valid_edits1
        
        word_edits2 = self.edits2(word)
        return word_edits2 & self.vocab

    def generate_phonetics_candidates(self, word: str) -> set:
        target_code = jellyfish.metaphone(word)
        phonetic_candidates = {w for w in self.vocab if jellyfish.metaphone(w) == target_code}
        return phonetic_candidates
    
    def generate_candidates(self, word: str) -> set:
        return self.generate_edit_candidates(word) | self.generate_phonetics_candidates(word)

    def get_ngram_probability(self, context: tuple, word: str) -> float:
        ngram = tuple(context)
        if ngram not in self.ngram_model:
            return EPS
        if word not in self.ngram_model[ngram]:
            return EPS
        return self.ngram_model.get(ngram, EPS)[word]
    
    def weighted_score(self, context, candidate, original):
        prob = self.get_ngram_probability(context, candidate)
        edit_penalty = 1 if candidate in self.edits1(original) else 2
        bonus = 1.2 if candidate in (self.edits1_keyboard(original) & self.vocab) else 1.0
        alpha = 0.9
        return prob * (alpha ** edit_penalty) * bonus

    def correct_sentence(self, sentence: str) -> str:
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

            best_candidate = max(candidates, key=lambda w: self.weighted_score(context, w, word))
            corrected_words.append(best_candidate)
        return ' '.join(corrected_words)


# usage

# ngram_model_path = r"models\trigram_model_shakespeare.pkl"
# spell_corrector = SpellCorrectorImproved(ngram_model_path, n=3)

# sentence = "to be or not to ne"
# corrected_sentence = spell_corrector.correct_sentence(sentence)
# print("  initial: ", sentence)
# print("corrected: ", corrected_sentence)
