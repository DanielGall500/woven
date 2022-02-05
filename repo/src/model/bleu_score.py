from sacrebleu.metrics import BLEU
import plotly.graph_objects as go
import numpy as np

"""
--WOVEN NMT Model BLEU Scorer--
Desc: Calculates the BLEU scores for the WOVEN NMT model using
the WMT20 English to German reference set.
For our tests, we will use sacreBLEU, which provides "hassle-free
computation of shareable, comparable, and reproducible BLEU scores."

The precision of an n-gram (ie a group of tokens) is interpreted as
the words covered by our candidate in the reference sets divided by
the total number of words in the candidate. In summary:
Precision = Words Covered in Ref / Total Words in Candidate
A brevity penalty is applied to penalise candidate translations which
deviate from the length of any their respective references references.

Results: The model achieves a BLEU score of 31.
This can be interpreted as "understandable to good translations".
Note that more reference sets per translation would likely improve
this further.

Please refer to the BLEU Score report for further details.
"""

class BLEUScorer(BLEU):
	def __init__(self, candidates: list[str], references: list[list[str]]):
		BLEU.__init__(self)
		self.c = candidates
		self.r = references
		self.num_samples = len(candidates)
		self.scr = self.corpus_score(candidates, references)

	def get_precisions(self):
		return self.scr.precisions

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

	#References must be wrapped as there can be multiple
	#references (2D) for each candidate (1D)
	references = [read(path_to_references)]
	candidates = read(path_to_candidates)

	#Compute BLEU Score
	scorer = BLEUScorer(candidates, references)
	scorer.print_results()

	#Plot the N-Gram Results
	grams = ["1-Gram", "2-Gram", "3-Gram", "4-Gram"]
	precisions = scorer.get_precisions()

	fig = go.Figure([go.Bar(x=grams, y=precisions)])
	fig.update_layout(title_text="BLEU N-Gram Precision Scores",
		autosize=False,
	    width=500,
	    height=500,)
	fig.update_xaxes(title="N-Gram")
	fig.update_yaxes(title="Precision (%)")
	fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
	fig.show()

if __name__ == "__main__":
	main()