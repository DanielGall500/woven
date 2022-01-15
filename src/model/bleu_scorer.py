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
from nmt import NMTModel
import numpy as np

class BLEUScorer(BLEU):
	def __init__(self, candidates: list[str], references: list[list[str]]):
		BLEU.__init__(self)
		self.c = candidates
		self.r = references
		self.num_samples = len(candidates)
		self.scr = self.corpus_score(candidates, references)

	def print_results(self):
		bleu_info = self.scr
		n = len(bleu_info.precisions)

		print("---NMT Model Evaluation---")
		print("Number of Input Sentences: ", self.num_samples)

		print("\nN-Gram Precision Scores")
		for ngram in range(n):
			correct = bleu_info.counts[ngram]
			total = bleu_info.totals[ngram]
			precision = bleu_info.precisions[ngram]

			print("{}-Gram Correct: {}".format(ngram+1, correct))
			print("{}-Gram Total: {}".format(ngram+1, total))
			print("{}-Gram Precision: {:.2f}\n".format(ngram+1, precision))

		print("Brevity Penalty: {:.2f}".format(bleu_info.bp))
		print("\nBLEU Score: {:.2f}".format(bleu_info.score))
		print("----------------")

def write(translations: list[str], path: str):
	with open(path, 'w', encoding='utf-8') as w:
		for t in translations:
			w.write(t + "\n")

def read(path: str, with_strip=False):
	with open(path, 'r', encoding='utf-8') as r:
		if with_strip:
			return [t.strip() for t in r]
		else:
			return r.readlines()

def main():
	path_to_source = 'data/wmt20_src_set.txt'
	path_to_candidates = 'data/wmt20_cand_set.txt'
	path_to_references = 'data/wmt20_ref_set.txt'

	samples_per_batch = 50

	#Translate
	source = read(path_to_source, with_strip=True)

	model = NMTModel('en','de')
	candidates = model.translate_in_batches(source, \
		samples_per_batch=samples_per_batch)

	#Save
	write(candidates, path_to_candidates)

	#References must be wrapped as there can be multiple
	#references (2D) for each candidate (1D)
	references = [read(path_to_references)]
	candidates = read(path_to_candidates)

	#Compute BLEU Score
	scorer = BLEUScorer(candidates, references)
	scorer.print_results()

if __name__ == "__main__":
	main()