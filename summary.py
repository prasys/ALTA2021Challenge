import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from summarizer import Summarizer,TransformerSummarizer
from transformers import *
import pandas as pd

def summarizeText(text):
    return(bert_model(text,use_first=False))
    
ModelName = "healx/gpt-2-pubmed-large"
# custom_config = AutoConfig.from_pretrained(ModelName)
# custom_config.output_hidden_states=True
# custom_tokenizer = AutoTokenizer.from_pretrained(ModelName)
# custom_model = AutoModel.from_pretrained(ModelName, config=custom_config)
        
##bert_model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
bert_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="healx/gpt-2-pubmed-large")
#bert_model = TransformerSummarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
train_df = pd.read_pickle('trainMinA.h5')
eval_df = pd.read_pickle('devMinA.h5')

train_df['text'] = train_df['text'].apply(summarizeText)
eval_df['text'] = eval_df['text'].apply(summarizeText)

train_df.to_pickle('trainMinB2.h5')
eval_df.to_pickle('devMinB2.h5')
train_df.to_csv("trainMinA2562.csv")
eval_df.to_csv("devMinA2562.csv")


# model(body2)
# bert_summary = ''.join(bert_model(body, min_length=60))
# print(bert_summary)