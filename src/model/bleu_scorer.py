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

class BLEUDataManager:
	def __init__(self, path_to_candidates, path_to_references):
		self.path_to_candidates = path_to_candidates
		self.path_to_references = path_to_references

	def write_candidates(self, candidates: list[str]):
		with open(self.path_to_candidates, 'r', encoding='utf-8') as w:
			for translation in candidates:
				w.write(translation + "\n")

	def read_candidates(self, n=-1):
		cands = self._read(self.path_to_candidates)

		if n > 0:
			return cands[:n]
		return cands

	def read_references(self, n=-1):
		refs = self._read(self.path_to_references)

		if n > 0:
			return refs[:n]
		return refs

	def _read(self, path):
		with open(path, 'r', encoding='utf-8') as r:
			return r.readlines()

def translate_set(model, in_path, out_path, stop_at_index=None):
	#with open(in_path, 'r', encoding='utf-8') as src_reader:
	#	source = [t.strip() for t in src_reader]

	source = read_in_samples(in_path, with_strip=True)

	#Limit to a certain number of translations
	if stop_at_index != None:
		source = source[:stop_at_index]

	num_samples = len(source)

	source = split_into_batches(source, samples_per_batch=50)
	candidates = []

	for i,translation_set in enumerate(source):
		print("Set {}".format(i))
		print("Translating...")

		#Candidates are potentially good or bad translations
		#made by the model
		candidate_set = model.translate(translation_set)

		print("Complete.\n")

		#Store out set of 50 with the others
		candidates.append(candidate_set)

	candidates = np.reshape(candidates, (-1,)).tolist()

	#Output translation of source to candidate file
	with open(out_path, 'w', encoding='utf-8') as cand_writer:
		for translation in candidates:
			cand_writer.write(translation + "\n")

def split_into_batches(samples: list[str], samples_per_batch=25):
	return np.reshape(samples, (-1, samples_per_batch)).tolist()

def read_in_samples(path, with_strip=False):
	with open(path, 'r', encoding='utf-8') as reader:
		if with_strip:
			return [t.strip() for t in reader]
		else:
			return reader.readlines()

def main():
	path_to_english_source = 'wmt20_src_set.txt'
	path_to_candidates = 'wmt20_cand_set.txt'
	path_to_references = 'wmt20_ref_set.txt'

	sample_size = 50

	#translate_set(NMTModel('en','de'), source_path, candidate_path, stop_at_index=50)

	bleu_data = BLEUDataManager(path_to_candidates, path_to_references)

	#References must be wrapped as dims mean more reference set
	references = [bleu_data.read_references(n=sample_size)]
	candidates = bleu_data.read_candidates()

	scorer = BLEUScorer(candidates, references)
	scorer.print_results()

if __name__ == "__main__":
	main()