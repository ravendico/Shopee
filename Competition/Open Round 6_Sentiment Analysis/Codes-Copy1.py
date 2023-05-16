# !pip install nltk

import torch
import torch.optim as optim

import random 

# fastai
from fastai import *
from fastai.text import *
from fastai.callbacks import *

# transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig

import fastai
import transformers

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.model_selection import cross_validate
    
import matplotlib.pyplot as plt

import pandas as pd
import os
import re
import nltk
import emoji
from scripts import *

if __name__ == '__main__':
	train = pd.read_csv('D:\\Shopee\\Open Round 6_Sentiment Analysis\\train.csv')
	test = pd.read_csv('D:\\Shopee\\Open Round 6_Sentiment Analysis\\test.csv')

	pd.options.display.max_colwidth = 1000
	pd.options.display.max_rows = 1000

	MODEL_CLASSES = {
    		'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    		'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    		'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    		'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    		'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)
		}

	# Parameters
	seed = 42
	use_fp16 = False
	bs = 16

	model_type = 'roberta'
	pretrained_model_name = 'roberta-base'

	# model_type = 'bert'
	# pretrained_model_name='bert-base-uncased'

	# model_type = 'distilbert'
	# pretrained_model_name = 'distilbert-base-uncased'

	#model_type = 'xlm'
	#pretrained_model_name = 'xlm-clm-enfr-1024'

	# model_type = 'xlnet'
	# pretrained_model_name = 'xlnet-base-cased'

	model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]

	def seed_all(seed_value):
    		random.seed(seed_value) # Python
				np.random.seed(seed_value) # cpu vars
    		torch.manual_seed(seed_value) # cpu  vars
    
    		if torch.cuda.is_available(): 
        		torch.cuda.manual_seed(seed_value)
        		torch.cuda.manual_seed_all(seed_value) # gpu vars
        		torch.backends.cudnn.deterministic = True  #needed
        		torch.backends.cudnn.benchmark = False
	seed_all()
    
	class TransformersBaseTokenizer(BaseTokenizer):
		"""Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
		def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
			self._pretrained_tokenizer = pretrained_tokenizer
			self.max_seq_len = pretrained_tokenizer.max_len
			self.model_type = model_type

		def __call__(self, *args, **kwargs): 
			return self

		def tokenizer(self, t:str) -> List[str]:
			"""Limits the maximum sequence length and add the spesial tokens"""
			CLS = self._pretrained_tokenizer.cls_token
			SEP = self._pretrained_tokenizer.sep_token
			if self.model_type in ['roberta']:
				tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
				tokens = [CLS] + tokens + [SEP]
			else:
				tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
				if self.model_type in ['xlnet']:
					tokens = tokens + [SEP] +  [CLS]
				else:
					tokens = [CLS] + tokens + [SEP]
			return tokens

	transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
	transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
	fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])

	class TransformersVocab(Vocab):
		def __init__(self, tokenizer: PreTrainedTokenizer):
        		super(TransformersVocab, self).__init__(itos = [])
        		self.tokenizer = tokenizer
    
    		def numericalize(self, t:Collection[str]) -> List[int]:
        		"Convert a list of tokens `t` to their ids."
        		return self.tokenizer.convert_tokens_to_ids(t)
        		#return self.tokenizer.encode(t)

    		def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        		"Convert a list of `nums` to their tokens."
        		nums = np.array(nums).tolist()
        		return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
    
    		def __getstate__(self):
        		return {'itos':self.itos, 'tokenizer':self.tokenizer}

    		def __setstate__(self, state:dict):
        		self.itos = state['itos']
        		self.tokenizer = state['tokenizer']
        		self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

	transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
	numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)

	tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)

	transformer_processor = [tokenize_processor, numericalize_processor]

	pad_first = bool(model_type in ['xlnet'])
	pad_idx = transformer_tokenizer.pad_token_id

	defaults.cpus=1

	torch.cuda.get_device_name(0)

	databunch = (TextList.from_df(train.dropna(subset=["review_clean"]), 
                              cols='review_clean', 
                              processor=transformer_processor)
             .split_by_rand_pct(0.1,seed=seed)
             .label_from_df(cols= 'rating')
             .add_test(test)
             .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx, num_workers=0))

	print('[CLS] token :', transformer_tokenizer.cls_token)
	print('[SEP] token :', transformer_tokenizer.sep_token)
	print('[PAD] token :', transformer_tokenizer.pad_token)
	databunch.show_batch()
