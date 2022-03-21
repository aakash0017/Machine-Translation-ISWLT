# -*- coding: utf-8 -*-
"""BERT2BERT_french.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dia9WD9HtWask3fGHPJppsHF5tWPWmyG
"""

# !pip install transformers datasets sentencepiece s3fs git-python==1.0.3 rouge_score sacrebleu boto3 --quiet

import boto3
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, EncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from s3fs import S3FileSystem
import warnings
warnings.simplefilter(action='ignore', category=Warning)

dataset = load_dataset('enimai/must_c_french')

model_checkpoint = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenization
encoder_max_length=128
decoder_max_length=128
source_lang = "en"
target_lang = "fr"

def add_verbosity(inputs: list, targets: list, test=False):
  processed_inputs = []
  for input, target in zip(inputs, targets):
    if test:
      input = 'normal' + ' ' + input
    else:
      ts_ratio = len(target)/len(input)
      if ts_ratio < 0.97:
        prefix = 'short'
        input = prefix + ' ' + input
      elif ts_ratio >= 0.97 or ts_ratio <= 1.05:
        prefix = 'normal'
        input = prefix + ' ' + input
      else:
        prefix = 'long'
        input = prefix + ' ' + input
    processed_inputs.append(input)
  return processed_inputs

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  source, target = batch[source_lang], batch[target_lang]
  # source = add_verbosity(source, target)
  source = add_verbosity(source, target)
  inputs = tokenizer(source, padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(target, padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  # batch["decoder_input_ids"] = outputs.input_ids
  # batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

def process_data_to_model_inputs_test(batch):
  # tokenize the inputs and labels
  source, target = batch[source_lang], batch[target_lang]
  # source = add_verbosity(source, target)
  source = add_verbosity(source, target, test=True)
  inputs = tokenizer(source, padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(target, padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  # batch["decoder_input_ids"] = outputs.input_ids
  # batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

# configure s3
# s3 = S3FileSystem(key='AKIA4QB2WTN57SCTNAGG', secret='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')

# tokenize validation data
print("tokenize validation data")
column_names = dataset['validation'].column_names
tokenized_val_data = dataset['validation'].select(range(16)).map(
    process_data_to_model_inputs,
    batched=True,
    remove_columns = column_names,
    desc="tokenizing validation data"
)
# tokenized_val_data.save_to_disk('s3://mtacl/tokenized_data_enc_dec_french_xlmr/validation', fs=s3)

# tokenize train data
print("tokenize train data")
column_names = dataset['train'].column_names
tokenized_train_data = dataset['train'].select(range(32)).map(
    process_data_to_model_inputs,
    batched=True,
    remove_columns = column_names,
    desc="tokenizing train data"
)
# tokenized_train_data.save_to_disk('s3://mtacl/tokenized_data_enc_dec_french_xlmr/train', fs=s3)

# tokenize test data
print("tokenize test data")
# changes test=True under add_verbosity function call in process_data_to_model_inputs
column_names = dataset['test'].column_names
tokenized_test_data = dataset['test'].map(
    process_data_to_model_inputs,
    batched=True,
    remove_columns = column_names,
    desc="tokenizing test data"
)

tokenized_train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)
tokenized_val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)
tokenized_test_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)

# Initialize Encoder-Decoder Model using XLM-Roberta checkpoints
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(model_checkpoint, model_checkpoint)

# set special tokens
bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
bert2bert.config.eos_token_id = tokenizer.eos_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id

# sensible parameters for beam search
bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
bert2bert.config.max_length = 142
bert2bert.config.min_length = 56
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4

# load rouge for validation
rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

batch_size=2
# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    logging_steps=2,  # set to 1000 for full training
    save_steps=16,  # set to 500 for full training
    eval_steps=4,  # set to 8000 for full training
    warmup_steps=1,  # set to 2000 for full training
    max_steps=16, # delete for full training
    overwrite_output_dir=True,
    save_total_limit=3,
    fp16=True, 
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=bert2bert,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_val_data,
)
trainer.train()

