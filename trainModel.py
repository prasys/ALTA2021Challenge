from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import pickle5 as pickle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import sklearn


def calcMicroF1(trueP,falseP):
    """[Caclulates MicroF1 scores using Sklearn implementation]

    Args:
        trueP ([list]): [True Positives]
        falseP ([list]): [Predicted Positives]

    Returns:
        [float]: [Micro F1 Score]
    """
    return sklearn.metrics.f1_score(trueP,falseP,average='micro')

def calcMacroF1(trueP,falseP):
    return sklearn.metrics.f1_score(trueP,falseP,average='macro')

def calcMacroRecall(trueP,falseP):
    return sklearn.metrics.recall_score(trueP,falseP,average='macro')

def calcMicroRecall(trueP,falseP):
    return sklearn.metrics.recall_score(trueP,falseP,average='macro')

def calcAccScore(trueP,falseP):
    return sklearn.metrics.balanced_accuracy_score(trueP,falseP)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_df = pd.read_pickle('train.h5')

# Preparing eval data
eval_df = pd.read_pickle('dev.h5')

# Optional model configuration
model_args = ClassificationArgs()
model_args.reprocess_input_data = False
model_args.overwrite_output_dir = True
model_args.use_multiprocessing = True
model_args.num_train_epochs=5
model_args.use_early_stopping= True
model_args.do_lower_case = True
model_args.max_seq_length = 500
model_args.best_model_dir = "output/best_bio"
model_args.train_batch_size = 32


# Create a ClassificationModel
model = ClassificationModel(
    'bert',
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    num_labels=3,
    args=model_args,
    weight=[0.313,0.460,0.227]
) 

# Train the model
model.train_model(train_df,output_dir='output/best_bio')

# model = ClassificationModel(
#     "bert", "/data/pradeesh/ALTA2021Challenge/output/best_bio"
# )

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df,micF1=calcMicroF1,macroF1=calcMacroF1,macroRecall=calcMacroRecall,microRecall=calcMicroRecall,acc=calcAccScore)
print(result)

