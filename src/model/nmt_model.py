"""
Phase I: Setting Up an English-German NMT Model

A Neural Machine Translation (NMT) model is the first step in the development cycle. 
Marian (Junczys-Dowmunt et al. 2018) is an efficient and open-source NMT framework written in pure
C++ and with minimal dependencies. Hugging Face provides a pre-trained Marian MT model which can be 
used and deployed in python.
"""
from transformers import MarianMTModel, MarianTokenizer

src_text = [">>deu<< Happy Christmas!"]

model_name = 'Helsinki-NLP/opus-mt-en-gem'
tokenizer = MarianTokenizer.from_pretrained(model_name)
print(tokenizer.supported_language_codes)

model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
decoded = [tokenizer.decode(t) for t in translated]
print(decoded)


