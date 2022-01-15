"""
Phase I: Setting Up an English-German NMT Model

A Neural Machine Translation (NMT) model is the first step in the development cycle. 
Marian (Junczys-Dowmunt et al. 2018) is an efficient and open-source NMT framework written in pure
C++ and with minimal dependencies. Hugging Face provides a pre-trained Marian MT model which can be 
used and deployed in python.
"""
from transformers import MarianMTModel, MarianTokenizer
import numpy as np

def split_into_batches(samples: list[str], samples_per_batch=25):
	n = len(samples)
	leftover = len(samples) % samples_per_batch
	samples = samples[:n-leftover]
	return np.reshape(samples, (-1, samples_per_batch)).tolist()

class NMTModel:
	def __init__(self, source, target):
		model_name = "Helsinki-NLP/opus-mt-{}-{}".format(source, target)

		#The tokenizer converts the text to a more NN-suitable
		#format, as it cannot take raw text.
		#Uses sentencepiece library
		self.tokenizer = MarianTokenizer.from_pretrained(model_name)

		#Load the model
		self.model = MarianMTModel.from_pretrained(model_name)

	def translate(self, src: str):
		#Encode: convert to format suitable for an NN
		encode_src = self.tokenizer(src, return_tensors="pt", padding=True)

		#Feed forward through NMT model
		translation = self.model.generate(**encode_src)

		#Decode: convert to regular string
		decode_target = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translation]
		return decode_target

	def translate_in_batches(self, src: list[str], samples_per_batch=25):
		minibatches = split_into_batches(src, samples_per_batch)
		num_batches = len(minibatches)
		fullbatch = []

		for i, minibatch in enumerate(minibatches):
			print("Batch {} of {}".format(i, num_batches))
			print("Translating...")

			#Candidates are potentially good or bad translations
			#made by the model
			translated_minibatch = self.translate(minibatch)

			print("Complete.\n")

			#Store out set of 50 with the others
			fullbatch.append(translated_minibatch)

		#Transform: 2D array of batches => 1D array of translations
		fullbatch = np.reshape(fullbatch, (-1,)).tolist()
		return fullbatch


