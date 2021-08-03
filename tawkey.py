import pandas as pd
from transformers import BertTokenizer
print('Loading BERT tokenizer...')

train_df = pd.read_pickle('trainMinAS.h5')
train_text = train_df['text']

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', do_lower_case=True)

max_len = 0

# For every sentence...
for sent in train_text:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)