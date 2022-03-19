## Needed for Cuda debugging as Cuda errors can be very cryptic
import csv, requests, json
from itertools import compress
import warnings, sys
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import transformers
import datasets
from datasets import list_datasets, load_dataset, list_metrics, load_metric
import sentencepiece as spm

from transformers import AutoModel, AutoTokenizer, AutoConfig

import numpy as np
from s3fs import S3FileSystem
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from argparse import ArgumentParser
import os

# add verbosity variable to the input text
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

# preprocess the inputs - by tokenizing the inputs
def preprocess_function(examples):
    max_source_length = 128
    max_target_length = 128
    padding = False
    # prefix = ""
    tokenizer = transformer_tokenizer
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    inputs = add_verbosity(inputs, targets)
    # inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Define the Transaltion Module
class TranslationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 2):
        super().__init__()
   
        # Define the model
        self.model = transformer_model
        # Defining batch size of our data
        self.batch_size = batch_size
          
        # Defining num_workers
        self.num_workers = num_workers

        # Defining Tokenizers
        self.tokenizer = transformer_tokenizer

        # Define label pad token id
        self.label_pad_token_id = -100
        self.padding = True

    def setup(self, stage=None):
        # Loading the dataset
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data

    def custom_collate(self,features):
        ## Pad the Batched data
        label_name = "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
        
    def train_dataloader(self):
        #dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #return DataLoader(train_dataset, sampler=dist_sampler, batch_size=32)
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    def val_dataloader(self):
         return DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    def test_dataloader(self):
         return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)         

# Define the Marian Translation Model
class MarianForTranslation(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        
        self.model = transformer_model
        # extract transformer name
        transformer_name = self.model.name_or_path
        # extract AutoConfig, from which relevant parameters can be extracted.
        self.config = transformers.AutoConfig.from_pretrained(transformer_name)
        self.tokenizer = transformer_tokenizer
        
        self.encoder = self.model.get_encoder()
        self.decoder = self.model.get_decoder()
        
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = torch.nn.Linear(self.config.d_model, self.model.shared.num_embeddings, bias=False)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def forward(self, batch):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        decoder_input_ids = batch['decoder_input_ids'] if 'decoder_input_ids' in batch.keys() else None
        encoder_outputs = batch['encoder_outputs'] if 'encoder_outputs' in batch.keys() else None

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = self.shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs = encoder_outputs
        )
        
        # ------
        # TODO, check the decoder output and also connect the last element of the output to the classifier
        # ------
        
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        return lm_logits

# Define the Translation Pytorch Lightning Class
class Translation(pl.LightningModule):

    def __init__(self, learning_rate: float = 2e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = MarianForTranslation()
        self.tokenizer = transformer_tokenizer
        self.ignore_pad_token_for_loss = True      

    def training_step(self, batch, batch_nb):
        # batch
        labels = batch['labels']
        
        # fwd
        y_hat = self.model(batch)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(y_hat.view(-1, self.model.config.vocab_size), labels.view(-1))
        self.log_dict({'train_loss':masked_lm_loss}, prog_bar=True)

        return masked_lm_loss

    def validation_step(self, batch, batch_nb):
        # batch
        labels = batch['labels']

        # fwd
        y_hat = self.model(batch)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(y_hat.view(-1, self.model.config.vocab_size), labels.view(-1))

        metrics = self.compute_metrics([y_hat, labels])
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log_dict({'val_loss':masked_lm_loss, 'val_bleu':metrics['bleu'],'val_genlen':metrics['gen_len'] }, prog_bar=True)
        return masked_lm_loss

    def test_step(self, batch, batch_nb):
        # batch
        labels = batch['labels']
        
        # fwd
        y_hat = self.model(batch)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(y_hat.view(-1, self.model.config.vocab_size), labels.view(-1))
        
        metrics = self.compute_metrics([y_hat,labels])
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log_dict({'test_loss':masked_lm_loss, 'test_bleu':metrics['bleu'],'test_genlen':metrics['gen_len'] }, prog_bar=True)
        return masked_lm_loss
    
    def generate(self, batch, max_length: int = 20, num_beams: int = 1, num_return_sequences: int = 3):
        self.batch = batch.copy()
        self.batch['labels'] = None
        
        decoder_start_token_id = self.model.config.decoder_start_token_id
        
        self.batch['encoder_outputs'] = self.model.encoder(self.batch['input_ids'],attention_mask=self.batch['attention_mask'])
        # define the initial start of the decoder_input_ids token
        self.batch['decoder_input_ids'] = (
                    torch.ones((self.batch['input_ids'].shape[0], 1), dtype=torch.long) * decoder_start_token_id
                )
        
        if num_beams<=1:
          output = self.greedy_search(self.batch,max_length,num_beams)
        else:
          output = self.beam_search(self.batch,max_length,num_beams,num_return_sequences)
        
        return output
        
    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.hparams.learning_rate, eps=1e-08)
        # scheduler = {
        #   'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, steps_per_epoch=len(self.trainer.datamodule.train_dataloader()), epochs=self.hparams.max_epochs),
        #   'interval': 'step'  # called after each training step
        # } 
        #scheduler = StepLR(optimizer, step_size=1, gamma=0.2)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=1e-4, cycle_momentum=False,step_size_up=300)
        #scheduler = ReduceLROnPlateau(optimizer, patience=0, factor=0.2)
        #self.sched = scheduler
        #self.optim = optimizer
        #return [optimizer], [scheduler]
        return [optimizer]
        
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        #parser.add_argument('--drop_prob', default=0.2, type=float)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'train_val_data'), type=str)

        # training params (opt)
        parser.add_argument('--learning_rate', default=2e-5, type=float, help = "type (default: %(default)f)")
        return parser
    
    def greedy_search(self,batch, max_length, num_beams):
        from transformers import StoppingCriteriaList, MaxLengthCriteria
        pad_token_id = self.model.config.pad_token_id
        bos_token_id = self.model.config.bos_token_id
        eos_token_id = self.model.config.eos_token_id
        decoder_input_ids = batch['decoder_input_ids']
        scores = ()

        # instantiate stopping criteria
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length),])

        # keep track of which sequences are already finished
        unfinished_sequences = decoder_input_ids.new(decoder_input_ids.shape[0]).fill_(1)
        cur_len = decoder_input_ids.shape[-1]

        while True:
          lm_logits = self.model(batch)
          next_tokens_logits = lm_logits[:, -1, :]
      
          # argmax
          next_tokens = torch.argmax(next_tokens_logits, dim=-1)

          # finished sentences should have their next token be a padding token
          if eos_token_id is not None:
              assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
              next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

          # update generated ids, model inputs, and length for next step
          decoder_input_ids = torch.cat([decoder_input_ids, next_tokens[:, None]], dim=-1)
          cur_len = cur_len + 1

          # if eos_token was found in one sentence, set sentence to finished
          if eos_token_id is not None:
              unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

          # stop when each sentence is finished, or if we exceed the maximum length
          if unfinished_sequences.max() == 0 or stopping_criteria(decoder_input_ids, scores):
            break

          batch['decoder_input_ids'] = decoder_input_ids
       
        return decoder_input_ids

    def beam_search(self, batch, max_length, num_beams, num_return_sequences):
        from transformers import BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList, MaxLengthCriteria
        ## Intiate token ids
        pad_token_id = self.model.config.pad_token_id
        bos_token_id = self.model.config.bos_token_id
        eos_token_id = self.model.config.eos_token_id
        decoder_input_ids = batch['decoder_input_ids']
        scores = ()

        ### Expand the decoder, attention_mask, encoder_outputs to num_beams
        expand_size = num_beams
        expanded_return_idx = (
            torch.arange(decoder_input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1)
        )
        batch['decoder_input_ids'] = batch['decoder_input_ids'].index_select(0, expanded_return_idx)

        batch['attention_mask'] = batch['attention_mask'].index_select(0, expanded_return_idx)

        batch['encoder_outputs']["last_hidden_state"] = \
              batch['encoder_outputs'].last_hidden_state.index_select(0, expanded_return_idx)
        
        ### reassign decoder input ids to expanded batch variable
        decoder_input_ids = batch['decoder_input_ids']
        ## Initialize Beam Scorer variables
        batch_size = batch['input_ids'].shape[0]
        length_penalty = self.model.config.length_penalty
        early_stopping = self.model.config.early_stopping

        beam_scorer = BeamSearchScorer(
                        batch_size=batch_size,
                        num_beams=num_beams,
                        device=self.model.device,
                        length_penalty=length_penalty,
                        do_early_stopping=early_stopping,
                        num_beam_hyps_to_keep=num_return_sequences,
                    )
        # instantiate stopping criteria
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length),])
        # instantiate logits processors
        logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(5, eos_token_id=eos_token_id),])
        
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        batch_beam_size, cur_len = decoder_input_ids.shape
        
        assert (
                    num_beams * batch_size == batch_beam_size
                ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=decoder_input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
          lm_logits = self.model(batch)
          next_token_logits = lm_logits[:, -1, :]
          # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
          # cannot be generated both before and after the `nn.functional.log_softmax` operation.
          next_token_logits[:, self.model.config.pad_token_id] = float("-inf")  # never predict pad token.
          
          next_token_scores = torch.nn.functional.log_softmax(
                          next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
          ## This forces the sequence not to have end of sentence token until the minimum length tokens are generated
          ## The EOS token id is 1 in this model and will make that index equal to -infinity
          next_token_scores = logits_processor(decoder_input_ids, next_token_scores)
          next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

          # reshape for beam search
          vocab_size = next_token_scores.shape[-1]
          next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

          next_token_scores, next_tokens = torch.topk(
              next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
          )

          scores += (next_token_scores,)
          # Since the logits gets repeated num_beam times, this gives whether the token is in the first, second or third beam
          # if num_beams was 3. Took some time to understand this. 
          next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
          next_tokens = next_tokens % vocab_size

          # stateless
          beam_outputs = beam_scorer.process(
              decoder_input_ids,
              next_token_scores,
              next_tokens,
              next_indices,
              pad_token_id=pad_token_id,
              eos_token_id=eos_token_id,
          )
          beam_scores = beam_outputs["next_beam_scores"]
          beam_next_tokens = beam_outputs["next_beam_tokens"]
          beam_idx = beam_outputs["next_beam_indices"]

          decoder_input_ids = torch.cat([decoder_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
          # increase cur_len
          cur_len = cur_len + 1

          if beam_scorer.is_done or stopping_criteria(decoder_input_ids, scores):
            break
          batch['decoder_input_ids'] = decoder_input_ids
        
        sequence_outputs = beam_scorer.finalize(
            decoder_input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
          )
        return sequence_outputs
    
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = torch.argmax(preds, 2) 
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [torch.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean([tensor.cpu() for tensor in prediction_lens])
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    # Load a metric
    metric = load_metric("sacrebleu")

    # Define the Pre-Trained Models
    max_source_length = 128
    max_target_length = 128
    source_lang = "en"
    target_lang = "de"
    model_checkpoint = 'Helsinki-NLP/opus-mt-en-de'
    transformer_model = transformers.MarianModel.from_pretrained(model_checkpoint)
    transformer_tokenizer = transformers.MarianTokenizer.from_pretrained(model_checkpoint)
    transformer_config = transformers.MarianConfig.from_pretrained(model_checkpoint)
    print(hasattr(transformer_model, "prepare_decoder_input_ids_from_labels"))

    # extract data from remote repo  
    s3 = S3FileSystem(key='AKIA4QB2WTN57SCTNAGG', secret='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')
    train_data = datasets.load_from_disk('s3://mtacl/tokenized_data/train_ver', fs=s3)
    test_data = datasets.load_from_disk('s3://mtacl/tokenized_data/test_ver', fs=s3)
    val_data = datasets.load_from_disk('s3://mtacl/tokenized_data/validation_ver', fs=s3)

    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.getcwd()
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)

    # each LightningModule defines arguments relevant to it
    parser = Translation.add_model_specific_args(parent_parser,root_dir)

    parser.set_defaults(
        #profiler='simple',
        deterministic=True,
        max_epochs=1,
        gpus=1,
        limit_train_batches=0.5,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        distributed_backend=None,
        fast_dev_run=False,
        model_load=False,
        model_name='best_model',
    )

    args, extra = parser.parse_known_args()
    print(args)

    """ Main training routine specific for this project. """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    if (vars(args)['model_load']):
        model = Translation.load_from_checkpoint(vars(args)['model_name'])
    else:  
        model = Translation(**vars(args))

    # ------------------------
    # 2 CALLBACKS of MODEL
    # ------------------------

    # callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=3,
        verbose=True,
        mode='min',
        strict=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        #dirpath='my/path/',
        filename='translat-gertoen-epoch{epoch:02d}-val_loss{val_loss:.2f}',
        auto_insert_metric_name=False
    )

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(args,
        callbacks=[early_stop,lr_monitor,checkpoint_callback]
        )    


    seed_everything(42, workers=True)
    translation_dm = TranslationDataModule()

    # Train the Model
    trainer.fit(model, translation_dm)
    trainer.test()
