from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import pickle5 as pickle


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_df = pd.read_pickle('train.h5')

# Preparing eval data
eval_df = pd.read_pickle('dev.h5')

# Optional model configuration
model_args = ClassificationArgs()
model_args.num_train_epochs=15
model_args.labels_list = ["A", "B","C"]

# Create a ClassificationModel
model = ClassificationModel(
    'bert',
    'jambo/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-renet',
    num_labels=3,
    args=model_args,
    weight=[0,0.5,1]
) 

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

