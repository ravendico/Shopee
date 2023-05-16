#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install nltk


# In[2]:


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


# In[3]:


import fastai
import transformers

import pandas as pd
import os
import re
import nltk
# !pip install emoji
import emoji
from scripts import *

if __name__ == '__main__':
    train = pd.read_csv('D:\\Shopee\\Open Round 6_Sentiment Analysis\\train.csv')
    test = pd.read_csv('D:\\Shopee\\Open Round 6_Sentiment Analysis\\test.csv')

    MODEL_CLASSES = {
        'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
        'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
        'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
        'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
        'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)
    }


    # In[8]:


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


    # In[9]:


    model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]


    # It is worth noting that in this case, we use the transformers library only for a multi-class text classification task. For that reason, this tutorial integrates only the transformer architectures that have a model for sequence classification implemented. These model types are :
    # 
    # BERT (from Google)
    # XLNet (from Google/CMU)
    # XLM (from Facebook)
    # RoBERTa (from Facebook)
    # DistilBERT (from HuggingFace)
    # However, if you want to go further - by implementing another type of model or NLP task - this tutorial still an excellent starter.

    # In[10]:


    def seed_all(seed_value):
        random.seed(seed_value) # Python
        np.random.seed(seed_value) # cpu vars
        torch.manual_seed(seed_value) # cpu  vars

        if torch.cuda.is_available(): 
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value) # gpu vars
            torch.backends.cudnn.deterministic = True  #needed
            torch.backends.cudnn.benchmark = False

    seed_all(seed)


    # In the fastai library, data pre-processing is done automatically during the creation of the DataBunch. As you will see in the DataBunch implementation, the tokenizer and numericalizer are passed in the processor argument under the following format :

    # In[11]:


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


    # In[12]:


    transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
    transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
    fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])


    # In this implementation, be carefull about 3 things :
    # 
    # As we are not using RNN, we have to limit the sequence length to the model input size.
    # Most of the models require special tokens placed at the beginning and end of the sequences.
    # Some models like RoBERTa require a space to start the input string. For those models, the encoding methods should be called with add_prefix_space set to True.
    # Below, you can find the resume of each pre-process requirement for the 5 model types used in this tutorial. You can also find this information on the HuggingFace documentation in each model section.
    # 
    # bert:       [CLS] + tokens + [SEP] + padding
    # 
    # roberta:    [CLS] + prefix_space + tokens + [SEP] + padding
    # 
    # distilbert: [CLS] + tokens + [SEP] + padding
    # 
    # xlm:        [CLS] + tokens + [SEP] + padding
    # 
    # xlnet:      padding + tokens + [SEP] + [CLS]
    # 
    # 
    # It is worth noting that we don't add padding in this part of the implementation.  As we will see later, fastai manage it automatically during the creation of the DataBunch.

    # In[13]:


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


    # Now that we have our custom tokenizer and numericalizer, we can create the custom processor. Notice we are passing the include_bos = False and include_eos = False options. This is because fastai adds its own special tokens by default which interferes with the [CLS] and [SEP] tokens added by our custom tokenizer.

    # In[14]:


    transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
    numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)

    tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)

    transformer_processor = [tokenize_processor, numericalize_processor]


    # Setting up the Databunch¶
    # 
    # 
    # For the DataBunch creation, you have to pay attention to set the processor argument to our new custom processor transformer_processor and manage correctly the padding.
    # 
    # As mentioned in the HuggingFace documentation, BERT, RoBERTa, XLM and DistilBERT are models with absolute position embeddings, so it's usually advised to pad the inputs on the right rather than the left. Regarding XLNET, it is a model with relative position embeddings, therefore, you can either pad the inputs on the right or on the left.

    # In[16]:


    pad_first = bool(model_type in ['xlnet'])
    pad_idx = transformer_tokenizer.pad_token_id


    # In[17]:


    defaults.cpus=1


    # In[18]:


    torch.cuda.get_device_name(0)


    # In[19]:


    databunch = (TextList.from_df(train.dropna(subset=["review_clean"]).sample(100_000), 
                                  cols='review_clean', 
                                  processor=transformer_processor)
                 .split_by_rand_pct(0.1,seed=seed)
                 .label_from_df(cols= 'rating')
                 .add_test(test)
                 .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx, num_workers=4))


    # In[ ]:


    print('[CLS] token :', transformer_tokenizer.cls_token)
    print('[SEP] token :', transformer_tokenizer.sep_token)
    print('[PAD] token :', transformer_tokenizer.pad_token)
    databunch.show_batch()

