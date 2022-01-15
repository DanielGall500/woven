"""
Phase I: Setting Up an English-German NMT Model

A Neural Machine Translation (NMT) model is the first step in the development cycle. 
Marian (Junczys-Dowmunt et al. 2018) is an efficient and open-source NMT framework written in pure
C++ and with minimal dependencies. Hugging Face provides a pre-trained Marian MT model which can be 
used and deployed in python.
"""
from transformers import MarianMTModel, MarianTokenizer

class NMTModel:
	def __init__(self, source, target):
		model_name = "Helsinki-NLP/opus-mt-{}-{}".format(source, target)

		#The tokenizer converts the text to a more NN-suitable
		#format, as it cannot take raw text.
		#Uses sentencepiece library
		self.tokenizer = MarianTokenizer.from_pretrained(model_name)

		#Load the model
		self.model = MarianMTModel.from_pretrained(model_name)

	def translate(self, src_text):
		#Encode: convert to format suitable for an NN
		encode_src = self.tokenizer(src_text, return_tensors="pt", padding=True)

		#Feed forward through NMT model
		translation = self.model.generate(**encode_src)

		#Decode: convert to regular string
		decode_target = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translation]
		return decode_target


