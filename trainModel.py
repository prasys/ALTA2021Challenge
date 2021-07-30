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
    return sklearn.metrics.recall_score(trueP,falseP,average='macro')

def calcAccScore(trueP,falseP):
    return sklearn.metrics.balanced_accuracy_score(trueP,falseP)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_df = pd.read_pickle('train3.h5')

# Preparing eval data
eval_df = pd.read_pickle('dev3.h5')

# Optional model configuration
model_args = ClassificationArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.use_multiprocessing = True
model_args.num_train_epochs=3
model_args.use_early_stopping= True
model_args.do_lower_case = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "mcc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 5
model_args.evaluate_during_training_steps = 1000
model_args.max_seq_length = 128
model_args.best_model_dir = "output/best_core3"
model_args.train_batch_size = 16
model_args.eval_batch_size = 16
model_args.sliding_window = False
model_args.save_steps = -1
model_args.special_tokens_list = ["[NUMBER]","[DIGIT]","[PHONE]","[URL]","[EMAIL]"]
model_args.adam_epsilon = 1e-8 #1e-8 is default
model_args.polynomial_decay_schedule_lr_end = 1e-7 # 1e-7 is default
model_args.scheduler = "cosine_schedule_with_warmup"
model_args.wandb_project = "alta2021"
model_args.wandb_kwargs = {'name': 'pubMedBERT'}
model_args.train_custom_parameters_only = False
model_args.custom_parameter_groups = [
    {
        "params": ["classifier.weight"],
        "lr": 1e-10,
    },
    {
        "params": ["classifier.bias"],
        "lr": 1e-3,
        "weight_decay": 0.0,
    },
]
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
        "num_train_epochs": {"values": [2, 3, 5]},
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

def train():
    # wandb.init()
    model = ClassificationModel(
        'bert',
        # '',
        'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
        num_labels=3,
        args=model_args,
        # sweep_config=wandb.config,
        weight=[1.06,0.723,1.47]
    )
    print(model.get_named_parameters())
    model.train_model(train_df,output_dir='output/best_core3')
    result, model_outputs, wrong_predictions = model.eval_model(eval_df,**eval_metrics)
    # wandb.join()

train()
# wandb.agent(sweep_id, train)

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

