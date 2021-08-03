import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import pickle5 as pickle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import sklearn
import argparse
import os,sys
import wandb

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
    return sklearn.metrics.recall_score(trueP,falseP,average='micro')

def calcAccScore(trueP,falseP):
    return sklearn.metrics.accuracy_score(trueP,falseP)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_df = pd.read_pickle('trainMinAS.h5')

# Preparing eval data
eval_df = pd.read_pickle('devMinAS.h5')
eval_df['labels'].replace({1: 1, 2: 1}, inplace=True)
train_df['labels'].replace({1: 1, 2: 1}, inplace=True)

# Optional model configuration
model_args = ClassificationArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.use_multiprocessing = True
model_args.num_train_epochs=1
model_args.use_early_stopping= True
model_args.do_lower_case = False
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "mcc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 5
model_args.evaluate_during_training_steps = 1000
model_args.learning_rate = 5e-5 # 4e-4
model_args.max_seq_length = 128
model_args.best_model_dir = "output/best_binary"
model_args.train_batch_size = 16
model_args.eval_batch_size = 16
model_args.sliding_window = False
model_args.save_steps = -1
# model_args.special_tokens_list = ["[NUMBER]","[DIGIT]","[PHONE]","[URL]","[EMAIL]"]
model_args.adam_epsilon = 1e-8 #1e-8 is default
model_args.polynomial_decay_schedule_lr_end = 1e-7 # 1e-7 is default
model_args.scheduler = "linear_schedule_with_warmup"
model_args.wandb_project = "alta2021"
model_args.wandb_kwargs = {'name': 'BERT Best'}
# model_args.train_custom_parameters_only = False
# model_args.custom_parameter_groups = [
#     {
#         "params": ["classifier.weight"],
#         "lr": 1e-10,
#     },
#     {
#         "params": ["classifier.bias"],
#         "lr": 1e-3,
#         "weight_decay": 0.0,
#     },
# ]
# model_args.custom_parameter_groups = [
#     {
#         "params": ["classifier.weight", "bert.encoder.layer.10.output.dense.weight"],
#         "lr": 1e-10,
#     }
# ]

# Doing sweep to check our optimisation 
sweep_config = {
    "method": "grid",  # grid, random
    "metric": {"name": "train_loss", "goal": "minimize"},
    "parameters": {
        "num_train_epochs": {"values": [3]},
        # "learning_rate": {"min": 5e-5, "max": 4e-4},
    },
}

sweep_id = wandb.sweep(sweep_config, project="alta2021")

eval_metrics = {
    "macroF1" : calcMacroF1,
    "microF1" : calcMicroF1,
    "macroRecall" : calcMacroRecall,
    "microRecall" : calcMicroRecall,
    "Acc" : calcAccScore,
}

#jambo/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-renet - {3: 0.7494356659142212}
#jambo/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-euadr -{3:0.7110609480812641}
#jambo/biobert-base-cased-v1.1-finetuned-euadr {3"" 0.7020316027088036"}
#jambo/biobert-base-cased-v1.1-finetuned-NCBI-disease
def train():
    wandb.init()
    model = ClassificationModel(
        'bert',
        # '',
        'jambo/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-renet',
        num_labels=2,
        args=model_args,
        # sweep_config=wandb.config,
        weight=[1.088,0.9248]
    )
    model.train_model(train_df,output_dir='output/best_binary')
    result, model_outputs, wrong_predictions = model.eval_model(eval_df,**eval_metrics)
    wandb.join()

#train()
wandb.agent(sweep_id, train)

# Train the model


# model = ClassificationModel(
#     "bert", "/data/pradeesh/ALTA2021Challenge/output/best_bio"
# )

# Evaluate the model
#print(result)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', default='trainset.txt', type=str)
#     parser.add_argument('--modelPath', default='/trainset/', type=str)
#     parser.add_argument('--load', default='output.h5', type=bool)
#     parser.add_argument('--cleanText', default=True, type=bool)
#     opt = parser.parse_args()
#     xmlPath = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + opt.XMLdir
#     docCollections = []
#     truthLabels = []
#     loadFileAndParse(opt.file,xmlPath,truthLabels,docCollections,opt.cleanText)
#     createAndSaveDataFrame(truthLabels,docCollections,opt.processedFile)

# if __name__ == '__main__':
#     main()

