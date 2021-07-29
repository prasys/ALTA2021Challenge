from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import pickle5 as pickle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_df = pd.read_pickle('train.h5')

# Preparing eval data
eval_df = pd.read_pickle('dev.h5')

# Optional model configuration
model_args = ClassificationArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.use_multiprocessing = True
model_args.num_train_epochs=4

# Create a ClassificationModel
model = ClassificationModel(
    'bert',
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    num_labels=3,
    args=model_args,
    weight=[0.313,0.460,0.227]
) 

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)
print("macro f1 score " + precision_recall_fscore_support(eval_df['labels'],model_outputs),average='macro')
print("micro f1 score " + precision_recall_fscore_support(eval_df['labels'],model_outputs),average='micro')
print(wrong_predictions)

