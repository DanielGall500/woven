from nmt import NMTModel
import numpy as np

"""
BLEU Batch Translate (Script)
Translates all source translations from WMT20
English-German translation set and writes them
to the appropriate candidate file.
"""

#Write translations to file
def write(translations: list[str], path: str):
	with open(path, 'w', encoding='utf-8') as w:
		for t in translations:
			w.write(t + "\n")

def main():
	path_to_source = 'data/wmt20_src_set.txt'
	path_to_candidates = 'data/wmt20_cand_set.txt'

	samples_per_batch = 50
	source = read(path_to_source, with_strip=True)

	model = NMTModel('en','de')
	candidates = model.translate_in_batches(source, \
		samples_per_batch=samples_per_batch)

	write(candidates, path_to_candidates)

if __name__ == "__main__":
	main()