# import libraries & dependencies
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
import numpy as np
from tqdm import tqdm 
import wandb

# for logging loss to wandb.ai
access_key = "" # enter wandb secret_accces_key
wandb.login(key=access_key)

# data processing
raw_datasets = load_dataset("enimai/must_c_french")
metric = load_metric("sacrebleu")

# dataset description
print(raw_datasets)

# pre-trained model checkpoints
train_model_checkpoints = "Helsinki-NLP/opus-mt-en-fr"

# load the MarianMT tokenizer
tokenizer = AutoTokenizer.from_pretrained(train_model_checkpoints)

def add_verbosity(input_list, target_list, test=False):
  """
  input: list of source & target sequences
  output: processed source sequence based on the calculated length ratios 
  """
  processed_input = []
  for input, target in zip(input_list, target_list):
    ts_ratio = len(target)/len(input)
    if test:
      prefix = "normal"
    else:
      if ts_ratio < 0.95:
        prefix = "short"
      elif ts_ratio >= 0.95 and ts_ratio <= 1.10:
        prefix = "normal"
      else:
        prefix = "long"
    input = prefix + " " + input
    processed_input.append(input)
  return processed_input

# preprocess MUST-C dataset
max_input_length = 128 
max_target_length = 128
source_lang = "en"
target_lang = "fr"
def preprocess_function(examples):
    inputs = examples["en"]
    targets = examples["fr"]
    inputs = add_verbosity(inputs, targets) # append appropriate prompts 
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# tokenize raw data
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# training procedure
model = AutoModelForSeq2SeqLM.from_pretrained(train_model_checkpoints)

batch_size = 16 # change batch-size according to GPU availability 
model_name = train_model_checkpoints.split("/")[-1]

# define training model arguments
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True    
)

# initialize data-collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
    
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# initialize the trainer module
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# train the model
trainer.train()
