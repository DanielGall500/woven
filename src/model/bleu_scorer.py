"""
Testing the MarianMT English-German NMT Model
Generate BLEU scores.

Naive BLEU (Bi-Lingual Evaluation Understudy)
Compares the NMT's translation (candidate) with
a group of existing translations written by humans (reference).

Computes precision (proportion of tokens in candidate 
that appear in references)

Penalises words that appear in the candidate more times 
than they do in any of the references. Only counts the number
of words in the references and doesn't count the others.

Range: Between zero, the worst, and one, the best.

BLEU = Words Covered in Ref / Total Words in Candidate

However, this BLEU score is not the actual one used in practice.
We must look to an derivative for a more effective score.

Two problems with this BLEU score is that it does not take into
account word order, and additionally applies a better score to
smaller translations that contain words present in the reference,
regardless of how many words should truly be in it.

For our tests, we will use sacreBLEU, which provides "hassle-free
computation of shareable, comparable, and reproducible BLEU scores."
"""
from sacrebleu.metrics import BLEU

references = [
		["The dog bit the man.", "It was not unexpected.", "The man bit him first."],
		["THe dog had bit the man.", "No one was surprised.", "The man had bitten the dog."],
	   ]

candidates = ["The dog bit the man.", "It wasn't surprising.", "The man had just bitten him."]

bleu = BLEU()
print(bleu.corpus_score(candidates, references))
print(bleu.get_signature())