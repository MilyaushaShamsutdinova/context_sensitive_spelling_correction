# Context sensitive spelling correction

This project implements a context-sensitive spelling corrector that fixes spelling mistakes in input text lines. By leveraging _N-gram language model_, the corrector not only considers the likelihood of words in isolation but also takes into account the surrounding context. The project aims to improve upon baseline approaches (such as _Norvig’s classic spell corrector_) by incorporating additional features that address common typographical errors.


## N-gram model training

Notebook for N-gram model training can be found [here](https://github.com/MilyaushaShamsutdinova/context_sensitive_spelling_correction/blob/main/src/n-gram_model_training.ipynb). Two datasets were used to train the trigram models:

* [karpathy/tiny_shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) is dataset containing 40,000 lines of Shakespeare's plays. It's small dataset, providing limited vocabulary. This dataset was used to train N-gram model for baseline spelling corrector.

* [bookcorpus/bookcorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) is a dataset consisting of the text of around 7,000 self-published books. Offered a richer vocabulary and context. N-gram model trained on this dataset was used for improved spelling corrector.


## Implementation details with justifications

Code for spelling corrector implementation is provided [here](https://github.com/MilyaushaShamsutdinova/context_sensitive_spelling_correction/blob/main/src/spelling_corrector.py). The solution is organized into two main implementations:

1. **class `SpellCorrector`** (baseline):

    * _Candidate generation:_ Uses standard edit operations (deletes, transposes, replaces, and inserts) to generate potential corrections for out-of-vocabulary words.

    * _Context-sensitive correction:_ For each word that is not found in the vocabulary, the algorithm calculates the conditional probability based on the trigram model. The candidate with the highest probability given the preceding words is selected as the correction.

2. **class `SpellCorrectorImproved`** (enhanced baseline):

    * _Keyboard layout consideration:_ An additional candidate generation method, `edits1_keyboard`, was introduced. This method uses the concept of keyboard neighbors to capture errors caused by mistyping, like hitting an adjacent key. 

    * _Phonetic similarity consideration:_ By using the metaphone algorithm via the **_jellyfish_** library, the corrector generates candidates that sound similar to the misspelled word. This helps capture errors resulting from similar-sounding words but might be several edits away.

    * _Weighted scoring function:_ Instead of selecting a candidate solely based on its trigram probability, a new weighted scoring function `weighted_score` was implemented. This function multiplies the language model’s probability by an exponential decay factor that penalizes candidates based on the number of edits required (e.g., a penalty of 1 for a single edit vs. 2 for a double edit). Moreover, it adds a bonus when the candidate appears in the keyboard-neighbor edits. This refinement—motivated by the noisy channel model—allows the system to favor corrections that are both contextually appropriate and plausibly resulting from common typing mistakes. Parameter tuning (setting alpha=0.9, bonus=1.2) provides a way to control the balance between contextual probability and edit penalties.


## Evaluation

Notebook for spelling corrector evaluation can be found [here](https://github.com/MilyaushaShamsutdinova/context_sensitive_spelling_correction/blob/main/src/evaluate.ipynb). I used subset of [vishnun/SpellGram](https://huggingface.co/datasets/vishnun/SpellGram) dataset for evaluation. It consist of lines with grammatical and spelling errors along with corrected lines.

The evaluation uses two primary metrics:

 - **Word Accuracy:** This is the percentage of words in the corrected text that match the intended, correctly spelled version. It tells how many words were fully corrected without error.

 - **Character Error Rate:** This measures the proportion of individual character errors relative to the total number of characters.


| Model Description   | Word Accuracy (%) | Character Error Rate (%) |
|------------------|---------------|----------------------|
| Baseline spelling corrector based on N-gram model trained on Shakespeare dataset (small) | 70.58        | 7.81       |
| Baseline spelling corrector based on N-gram model trained on BookCorpus dataset (large)  | 82.54        | 4.43       |
| Improved spelling corrector based on N-gram model trained on BookCorpus dataset (large)  | 81.43        | 5.35       |


These experiments demonstrate that a larger corpus can better capture natural language patterns, directly impacting the correctness of context-sensitive corrections. However, it is interesting that the improved Spell Corrector performs slightly worse than the baseline. I suspect the reasons could be twofold:

 - first, additional candidate generation noise: by incorporating extra candidate sources (like phonetic candidates), the model introduces more noise into the candidate pool, which sometimes leads to suboptimal correction choices. This problem could be resolved by using some model for reranking candidates.
 
 - Second, parameter tuning complexity: function `weighted_score` requires more delicate tuning of hyperparameters; even slight mismatches in these settings can degrade overall performance.
