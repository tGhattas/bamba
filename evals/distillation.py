import os
import random
from typing import Optional, Union
import torch
import torch.nn as nn
from torch.nn import DataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling, get_scheduler, MambaForCausalLM, TrainingArguments, Trainer
from accelerate import Accelerator, load_checkpoint_and_dispatch
from datasets import load_dataset
from torch.utils.data import DataLoader
from itertools import islice
from numpy import isnan
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    class MambaLMHeadModel: pass
from tqdm.auto import tqdm
import numpy as np
import argparse
from memory import MemoryTrace
from kl_div_loss import KLDivLoss
from uld_loss import ULDLoss
from pprint import pprint
from modified_tokenizer import ModifiedMambaTokenizerFactory
import time
from hf_trainer import KDTrainer
import wandb
from trl import SFTTrainer



logger = None
accelerator = None
hf_mamba_path = "state-spaces/mamba-790m-hf"
teacher_model_path = "meta-llama/Meta-Llama-3-8B"
tiny_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
pythia_14m_model_path = "EleutherAI/pythia-14m"
pythia_1B_model_path = "EleutherAI/pythia-1b"
pythia_28B_model_path = "EleutherAI/pythia-2.8b"
pythia_69B_model_path = "EleutherAI/pythia-6.9b"
teacher_model_path = pythia_28B_model_path 

# teacher_model_path = "mistralai/Mistral-7B-v0.3"

def get_teacher_model(path: str):
    model = AutoModelForCausalLM.from_pretrained(path)
    print_model_parameters("Teacher", model)
    pprint(model.config) if accelerator is None else accelerator.print(model.config)
    return model


def get_sanity_student_model(path: str=None):
    if path:
        model = AutoModelForCausalLM.from_pretrained(path)
    else:
        # reduce the number of parameters in the student model
        config = AutoConfig.from_pretrained(tiny_model_path)
        teacher_model_config = AutoConfig.from_pretrained(teacher_model_path)
        config.eos_token_id = teacher_model_config.eos_token_id
        config.bos_token_id = teacher_model_config.bos_token_id
        config.vocab_size = teacher_model_config.vocab_size
        config.num_hidden_layers = config.num_hidden_layers
        model = AutoModelForCausalLM.from_config(config)
    # print memory foorprint and number of parameters
    # adapt TinyLlama-1.1B to the teacher model
    if (accelerator and accelerator.is_main_process) or accelerator is None:
        print_model_parameters("Sanity Student", model)
        pprint(model.config)
    return model



# MAMBA student model
def get_mamba_model(path: str = None, gpu: int = None, set_teacher_embedding_size: bool = False):
    device = f'cuda{f":{gpu}" if gpu else ""}' if torch.cuda.is_available() else 'mps'
    teacher_model = get_teacher_model(teacher_model_path)
    param = next(teacher_model.parameters())
    teacher_dtype = param.dtype
    if path:
        mamba_student_model = smart_to(MambaForCausalLM.from_pretrained(path), device)
        config = mamba_student_model.config
        if set_teacher_embedding_size:
            config.vocab_size = teacher_model.config.vocab_size

    else:
        config = AutoConfig.from_pretrained(hf_mamba_path)
        config.vocab_size = teacher_model.config.vocab_size
        config.torch_dtype = teacher_dtype
        mamba_student_model = smart_to(MambaForCausalLM(config), device)
    if (accelerator and accelerator.is_main_process) or accelerator is None:
        print_model_parameters("MAMBA", mamba_student_model)
        pprint(config)
    return mamba_student_model



# Step 3: Knowledge Distillation

HF_PADDING_IGNORE = -100


def init_dataloader(batch_size: int, max_length: int, partition: str = "train", student_tokenizer: Optional[Union[str, AutoTokenizer]] = None, minimize_dataset: bool = False, return_dataloader: bool = True):

    dataset_path = "wikitext-2-v1"

    if student_tokenizer is not None:
        dataset = load_dataset("wikitext", dataset_path)
       
        student_tokenizer = AutoTokenizer.from_pretrained(student_tokenizer, use_fast=True) if isinstance(student_tokenizer, str) else student_tokenizer
        student_tokenizer.pad_token = student_tokenizer.eos_token
        def student_tokenize_function(examples):
            return smart_to(student_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"), "cuda" if torch.cuda.is_available() else "mps")
        
        student_tokenized_datasets = dataset.map(student_tokenize_function, batched=True, remove_columns=["text"])
        student_data_collator = DataCollatorForLanguageModeling(
            tokenizer=student_tokenizer,
            mlm=False,  # Set to True if using Masked Language Modeling
            pad_to_multiple_of=8  # Optional, can pad to the nearest multiple of 8 for efficiency
        )

        _tokenized_data = student_tokenized_datasets[partition] if not minimize_dataset else student_tokenized_datasets[partition].select(range(10))
        student_data_loader = DataLoader(_tokenized_data, batch_size=batch_size, collate_fn=student_data_collator) if return_dataloader else _tokenized_data
    
    
    dataset = load_dataset("wikitext", dataset_path)
    # Load the teacher tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, use_fast=True)
    # add padding token to the tokenizer
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    if student_tokenizer:
        student_tokenizer.pad_token = teacher_tokenizer.pad_token
    # Tokenize the dataset
    def teacher_tokenize_function(examples):
        return smart_to(teacher_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"), "cuda" if torch.cuda.is_available() else "mps")

    teacher_tokenized_datasets = dataset.map(teacher_tokenize_function, batched=True, remove_columns=["text"])

    teacher_data_collator = DataCollatorForLanguageModeling(
        tokenizer=teacher_tokenizer,
        mlm=False,  # Set to True if using Masked Language Modeling
        pad_to_multiple_of=8  # Optional, can pad to the nearest multiple of 8 for efficiency
    )
    tokenized_data = teacher_tokenized_datasets[partition] if not minimize_dataset else teacher_tokenized_datasets[partition].select(range(10))
    # Create the data loader
    teacher_data_loader = DataLoader(tokenized_data, batch_size=batch_size, collate_fn=teacher_data_collator) if return_dataloader else tokenized_data

    if student_tokenizer:
        return (teacher_data_loader, student_data_loader, teacher_tokenizer.pad_token_id) if return_dataloader else (teacher_data_loader, student_data_loader, teacher_tokenizer.pad_token_id, teacher_data_collator, student_data_collator)
    
    return (teacher_data_loader, teacher_tokenizer.pad_token_id) if return_dataloader else  (teacher_data_loader, teacher_tokenizer.pad_token_id, teacher_data_collator)



def print_model_parameters(model_name: str, model: Union[AutoModelForCausalLM, MambaLMHeadModel]):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    printF = pprint if accelerator is None else accelerator.print
    printF(f"Model: {model_name}")
    printF(f"Total Parameters: {total_params}")
    printF(f"Trainable Parameters: {trainable_params}")
    printF(f"Total Memory Footprint: {total_params * 4 / 1024 / 1024} MB")

def logits_to_tokens(logits):
    """Convert logits to token ids."""
    return torch.argmax(logits, dim=-1)

def distill_knowledge(teacher_model: AutoModelForCausalLM, student_model: Union[MambaLMHeadModel, AutoModelForCausalLM], optimizer: torch.optim.Optimizer,
                       batch_size: int, max_length: int,
                         limit: int=1000, epochs: int=5, load_chkpt: bool=False, model_path: str=None, gpu: int = None, accumulation_steps: int = 1,
                           modified_tokenizer: bool = False, use_teacher_tokenizer: bool = False, teacher_model_path: str = None, minimize_dataset: bool = False, unique_id: str = '', alpha: float = 0.5, temperature: float = 2.0):
    device = f'cuda{f":{gpu}" if gpu else ""}' if torch.cuda.is_available() else 'mps'
    printF = pprint if accelerator is None else accelerator.print

    if load_chkpt:
        student_model.load_state_dict(torch.load(model_path))

    first_batch = True
    log_interval = 10
    
    running_loss = 0
    running_distillation_loss = 0
    running_cross_entropy_loss = 0
    
    student_underlying_model = student_model.module if isinstance(student_model, DataParallel) else student_model
    teacher_underlying_model = teacher_model.module if isinstance(teacher_model, DataParallel) else teacher_model

    assert not (modified_tokenizer and use_teacher_tokenizer), "Both modified_tokenizer and use_teacher_tokenizer cannot be True at the same time"
    if modified_tokenizer:
        student_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, use_fast=True)
        tokenizer_factory = ModifiedMambaTokenizerFactory(student_tokenizer=student_tokenizer, teacher_tokenizer=teacher_tokenizer)
        student_tokenizer = tokenizer_factory.get_modified_tokenizer()
        student_underlying_model.resize_token_embeddings(len(student_tokenizer))
        dataloader = init_dataloader(batch_size, max_length, "train", student_tokenizer=student_tokenizer, minimize_dataset=minimize_dataset)
        printF(f"Student Model Vocab Size: {student_tokenizer.vocab_size}")
        printF(f"Teacher Model Vocab Size: {teacher_tokenizer.vocab_size}")
    elif use_teacher_tokenizer:
        student_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, use_fast=True)
        student_tokenizer.pad_token = student_tokenizer.eos_token
        student_underlying_model.resize_token_embeddings(teacher_underlying_model.config.vocab_size)
        dataloader = init_dataloader(batch_size, max_length, "train", student_tokenizer=student_tokenizer, minimize_dataset=minimize_dataset)
        printF("Using Teacher Tokenizer for student model")
    else:
        dataloader = init_dataloader(batch_size, max_length, "train", student_tokenizer=model_path, minimize_dataset=minimize_dataset)

    teacher_vocab_size = teacher_underlying_model.config.vocab_size
    student_vocab_size = student_underlying_model.config.vocab_size
    if teacher_vocab_size == student_vocab_size:
        loss_fn = KLDivLoss(reduction='mean', temperature=temperature, ignore_idx=HF_PADDING_IGNORE, distillation_loss_weight=alpha, using_acc=False)
        printF("-"*50)
        printF("Using KL Divergence Loss")
        printF("-"*50)
    else:
        loss_fn = ULDLoss(distillation_weight=alpha, crossentropy_weight=1-alpha, ignore_idx=HF_PADDING_IGNORE, teacher_temperature=temperature, student_temperature=temperature, skip_student_eos=True, skip_teacher_eos=True)
        printF("-"*50)
        printF("Using ULD Loss")
        printF("-"*50)
    

    if  model_path or modified_tokenizer or use_teacher_tokenizer:
        teacher_train_dataloader, student_train_dataloader, pad_token_id = dataloader
    else:
        teacher_train_dataloader, pad_token_id = dataloader

    eval_dataloader, _ = init_dataloader(batch_size, max_length, "test", minimize_dataset=minimize_dataset)
    lr_scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=int(0.05 * epochs * len(teacher_train_dataloader)), num_training_steps=epochs * len(teacher_train_dataloader))

    if accelerator is not None:
        teacher_train_dataloader, student_train_dataloader, eval_dataloader, student_model, teacher_model, optimizer, lr_scheduler = accelerator.prepare(teacher_train_dataloader, student_train_dataloader, eval_dataloader, student_model, teacher_model, optimizer, lr_scheduler)


    other_dataloader = teacher_train_dataloader if not model_path else student_train_dataloader
    steps_per_epoch = len(teacher_train_dataloader)

    if (accelerator is not None) or accelerator is None:
        printF("PRE TRAINING EVALS")
        # evaluate the teacher model
        evaluate(teacher_model, eval_dataloader=eval_dataloader, is_student=False, pad_token_id=pad_token_id)
        evaluate(student_model, eval_dataloader=eval_dataloader, is_student=True, pad_token_id=pad_token_id)

    for epoch in range(epochs):
        with MemoryTrace() as mem_trace:
            progress_bar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=steps_per_epoch//accumulation_steps,
                                 dynamic_ncols=True, disable=(accelerator is not None and not accelerator.is_local_main_process))
            ignored_count = 0
            for batch_idx, (teacher_batch, student_batch)  in enumerate(islice(zip(teacher_train_dataloader, other_dataloader), limit)):
                teacher_batched_input_ids = smart_to(teacher_batch['input_ids'], device)
                student_batched_input_ids = smart_to(student_batch['input_ids'], device)
                
                teacher_inputs = smart_to(teacher_batched_input_ids[:, :-1].contiguous(), device)
                student_inputs = smart_to(student_batched_input_ids[:, :-1].contiguous(), device)

                student_labels = smart_to(student_batched_input_ids[:, 1:].contiguous(), device)
                student_labels[student_labels == pad_token_id] = HF_PADDING_IGNORE

                teacher_labels = smart_to(teacher_batched_input_ids[:, 1:].contiguous(), device)
                teacher_labels[teacher_labels == pad_token_id] = HF_PADDING_IGNORE

                batched_attention_mask = smart_to(teacher_batch['attention_mask'], device)
                attention_mask = smart_to(batched_attention_mask[:, :-1].contiguous(), device)
                
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids=teacher_inputs,
                                                    attention_mask=attention_mask
                                                    )
                if isinstance(student_model, MambaLMHeadModel):
                    raise NotImplementedError("non HF Mamba model not supported")
                else:
                    student_outputs = student_model(input_ids=student_inputs,
                                                    attention_mask=attention_mask,
                                                    labels=student_labels)

                if first_batch:
                    printF = pprint if accelerator is None else accelerator.print
                    printF(f"Student logits shape: {student_outputs.logits.shape}")
                    printF(f"Teacher logits shape: {teacher_outputs.logits.shape}")
                    first_batch = False
                
                
                if (teacher_labels == pad_token_id).all() or (student_labels == pad_token_id).all():
                    ignored_count += 1
                    continue
                
                loss, student_label_loss, distillation_loss = loss_fn(student_outputs, teacher_outputs, student_labels, teacher_labels)
                student_label_loss = student_label_loss.mean() / accumulation_steps
                distillation_loss = distillation_loss.mean() / accumulation_steps
                loss = loss.mean() / accumulation_steps
                running_loss += loss.detach().float()
                running_distillation_loss += distillation_loss.detach().float()
                running_cross_entropy_loss += student_label_loss.detach().float()
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.zero_grad()
                    if accelerator is not None:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
                    # Gradient clipping
                    (torch.nn.utils if accelerator is None else  accelerator).clip_grad_norm_(student_model.parameters(), max_norm=0.5)
                    optimizer.step()
                    logger.log({"running_loss": running_loss.item()})
                    logger.log({"running_distillation_loss": running_distillation_loss.item()})
                    logger.log({"running_cross_entropy_loss": running_cross_entropy_loss.item()})
                    logger.log({"learning_rate": optimizer.param_groups[0]['lr']})

                    running_loss = 0
                    running_distillation_loss = 0
                    running_cross_entropy_loss = 0
                    progress_bar.update()

                # evaluate the student model every 4 log intervals
                if batch_idx % log_interval == 0:
                    # evaluate the student model
                    evaluate(student_model, eval_dataloader=eval_dataloader, is_student=True, pad_token_id=pad_token_id, gpu=gpu)
                    student_model.train()
                
                lr_scheduler.step()
                # print till the 4th after the decimal point
                progress_bar.set_description(f"Training Epoch: {epoch+1}/{epochs} | Step: {batch_idx}/{steps_per_epoch} | Average Training Loss: {loss.mean().item():.5f}")
            
            progress_bar.close()
            print(f"Epoch: {epoch} completed")
            print(f"Ignored {ignored_count} batches")
        
        print(mem_trace) if accelerator is None else print(f"process index: {accelerator.process_index}", mem_trace)
        
        # if os.path.exists("./checkpoints") == False:
        #     os.mkdir("./checkpoints")
        # if (accelerator is not None and accelerator.is_main_process) or accelerator is None:
        #     torch.save(student_model.state_dict(), f"./checkpoints/u{unique_id}_student_chkpt_epoch_{epoch}_type_{'mamba' if isinstance(student_model, MambaLMHeadModel) else 'transformer'}_max_length_{max_length}.pt")
    
    
    if (accelerator is not None) or accelerator is None:
        printF("POST TRAINING EVALS")
        evaluate(teacher_model, eval_dataloader=eval_dataloader, is_student=False, pad_token_id=pad_token_id, gpu=gpu)
        evaluate(student_model, eval_dataloader=eval_dataloader, is_student=True, pad_token_id=pad_token_id, gpu=gpu)


def finetune_teacher(unique_id: str, batch_size: int, max_length: int, minimize_dataset:bool, epochs:int, lr: float, optimizer: str, mixed_precision: bool, tf32: bool, teacher_model_path: str = teacher_model_path):
    # fine tune teacher model using hf trainer

    train_dataset, _, teacher_data_collator = init_dataloader(batch_size, max_length, "train", minimize_dataset=minimize_dataset, return_dataloader=False)
    test_dataset, _, _ = init_dataloader(batch_size, max_length, "validation", minimize_dataset=minimize_dataset, return_dataloader=False)
    model = smart_to(AutoModelForCausalLM.from_pretrained(teacher_model_path), "cuda" if torch.cuda.is_available() else "mps")
    name = f"u{unique_id}_finetuned_{epochs}_ep_{teacher_model_path}_optm{optimizer}_mp{mixed_precision}".replace('.','').replace('/','')
    training_args = TrainingArguments(
        output_dir="./hf-results",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=lr,
        report_to="wandb",  # Enable logging to wandb
        gradient_accumulation_steps=64,
        remove_unused_columns=False,
        fp16=mixed_precision,
        optim=optimizer,
        gradient_checkpointing=True, ###
        lr_scheduler_type="cosine",
        run_name=name,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=teacher_data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        max_seq_length=max_length,
    )

    # Train the model
    trainer.train()
    trainer.save_model(name)
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    logger.log(eval_results)

    print("Evaluation results:", eval_results)
    

def hf_train(unique_id: str, teacher_model: AutoModelForCausalLM, student_model: Union[MambaLMHeadModel, AutoModelForCausalLM], minimize_dataset: bool,
                batch_size: int, max_length: int, epochs: int, model_path: str, accumulation_steps: int, alpha: float, temperature: float,
                  learning_rate: float, mixed_precision: bool, optimizer: str, tf32: bool):
    train_dataset, _, teacher_data_collator = init_dataloader(batch_size, max_length, "train", minimize_dataset=minimize_dataset, return_dataloader=False)
    test_dataset, _, _ = init_dataloader(batch_size, max_length, "test", minimize_dataset=minimize_dataset, return_dataloader=False)
    name = f"u{unique_id}_hf_trained_student_{epochs}_epochs_{model_path}_optim{optimizer}_mp{mixed_precision}".replace('.','').replace('/','')
    training_args = TrainingArguments(
        output_dir="./hf-results",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=learning_rate,
        report_to="wandb",  # Enable logging to wandb
        gradient_accumulation_steps=accumulation_steps,
        remove_unused_columns=False,
        lr_scheduler="cosine",
        optim="adamw_hf",
        gradient_checkpointing=True,
        fp16=mixed_precision,
        tf32=tf32,
        run_name=name,
    )
    trainer = KDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        temperature=temperature,
        alfa=alpha,
        args=training_args,
        data_collator=teacher_data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        accelerator=accelerator
    )

    # Train the model
    trainer.train()
    # save the model
    trainer.save_model(name)
    # Evaluate the model
    eval_results = trainer.evaluate()
    printF = pprint if accelerator is None else accelerator.print
    printF("Evaluation results:", eval_results)
    



# Training Loop
def train(limit: int = 1000, batch_size: int = 4, max_length: int = 128, epochs: int = 5,
           learning_rate: float = 5e-5, load_chkpt: bool=False, load_hf_model: bool=False, model_path: str=None,
             is_mamba: bool=False, gpu: int = None, accumulation_steps: int = 1, use_modified_tokenizer: bool = False,
               use_teacher_tokenizer: bool = False, teacher_model_path: str = teacher_model_path,
                 minimize_dataset: bool = False, unique_id: str = '', alpha: float = 0.5, temperature: float = 2.0, hf_trainer: bool = False,
                   optimizer=None, mixed_precision: bool = False, tf32: bool = False):   
    # assert that if either load_chkpt or load_hf_model is True but not both
    assert not (load_chkpt and load_hf_model), "Both load_chkpt and load_hf_model cannot be True at the same time"
    device = f'cuda{f":{gpu}" if gpu else ""}' if torch.cuda.is_available() else 'mps'
    teacher_model = get_teacher_model(teacher_model_path)
    if accelerator is None:
        teacher_model = DataParallel(teacher_model)
    smart_to(teacher_model, device)
    
    teacher_model.eval()
    print_model_parameters(teacher_model_path, teacher_model)
    if load_hf_model:
        if not is_mamba:
            student_model = smart_to(AutoModelForCausalLM.from_pretrained(model_path), device)
        else:
            student_model = get_mamba_model(path=model_path, gpu=gpu)
    else:
        if not is_mamba:
            student_model = smart_to(get_sanity_student_model(), device)
        else:
            student_model = get_mamba_model(gpu=gpu)
    if accelerator is None:
        student_model = DataParallel(student_model)
    student_model.train()

    
    if hf_trainer:
        hf_train(teacher_model, student_model, optimizer, batch_size, max_length, epochs, model_path, accumulation_steps, alpha, temperature, learning_rate, mixed_precision, optimizer)
    else:
        optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
        distill_knowledge(teacher_model, student_model, optimizer, batch_size, max_length, limit=limit, epochs=epochs,
                        load_chkpt=load_chkpt, model_path=model_path, gpu=gpu, accumulation_steps=accumulation_steps,
                            modified_tokenizer=use_modified_tokenizer, use_teacher_tokenizer=use_teacher_tokenizer, teacher_model_path=teacher_model_path,
                                minimize_dataset=minimize_dataset, unique_id=unique_id, alpha=alpha, temperature=temperature)
    # save the student model
    if accelerator is None:
        (student_model.module if isinstance(student_model, DataParallel) else student_model).save_pretrained(f"u{unique_id}_{epochs}_lr_{learning_rate}_is_mamba_{is_mamba}_max_length_{max_length}_alfa_{alpha}_tmp_{temperature}_{model_path}")
    else:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(student_model)
        unwrapped_model.save_pretrained(f"u{unique_id}distr_{epochs}_lr_{learning_rate}_is_mamba_{is_mamba}_max_length_{max_length}_alfa_{alpha}_tmp_{temperature}_{model_path}")


# Evaluate the student model
def evaluate(model_or_path: Union[str, AutoModelForCausalLM, MambaLMHeadModel, MambaForCausalLM], gpu: int = None, eval_dataloader: DataLoader = None, pad_token_id: int = None, is_student: bool = True):
    device = f'cuda{f":{gpu}" if gpu else ""}' if torch.cuda.is_available() else 'mps'
    if accelerator is not None:
        accelerator.wait_for_everyone()
    # evaluate the student model using the test dataset
    if isinstance(model_or_path, str):
        student_model = smart_to(AutoModelForCausalLM.from_pretrained(model_or_path), device)
    else:
        student_model = model_or_path
    
    student_model.eval()
    if eval_dataloader is None:
        dataloader, pad_token_id = init_dataloader(8, 128, "test")
    else:
        dataloader, pad_token_id = eval_dataloader, pad_token_id
    
    # evalua using the test dataset
    running_loss = 0
    counter = 1
    ignored_count = 0
    # student_model_eval = student_model.module if isinstance(student_model, DataParallel) else student_model
    start = time.perf_counter()
    for batch in tqdm(dataloader):
        
        batched_input_ids = smart_to(batch['input_ids'], device)
        inputs = smart_to(batched_input_ids[:, :-1].contiguous(), device)
        labels = smart_to(batched_input_ids[:, 1:].contiguous(), device)
        # skip the batch if all the labels are pad tokens
        if (labels == pad_token_id).all():
            ignored_count += 1
            continue

        labels[labels == pad_token_id] = HF_PADDING_IGNORE

        batched_attention_mask = smart_to(batch['attention_mask'], device)
        attention_mask = smart_to(batched_attention_mask[:, :-1].contiguous(), device)
        with torch.no_grad():

            if isinstance(student_model, MambaLMHeadModel):
                student_outputs = smart_to(student_model(input_ids=inputs,
                                            ).logits, device)
            else:
                student_outputs = smart_to(student_model(input_ids=inputs,
                                                attention_mask=attention_mask,
                                                labels=labels
                                                ).logits, device)

        student_label_loss = nn.CrossEntropyLoss(ignore_index=HF_PADDING_IGNORE)(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))

        running_loss += student_label_loss.item()
        if isnan(running_loss):
            print("NaN loss detected")
            print(f"student_outputs: {student_outputs}")
            print(f"labels.view(-1): {labels.view(-1)}")
            print(f"inputs: {inputs}")
            print(f"origin of labels: {batched_input_ids}")
            raise ValueError("NaN loss detected")
        counter += 1
            
    duration = time.perf_counter() - start
    prefix = "student_" if is_student else "teacher_"
    logger.log({f"{prefix}test_loss": running_loss / counter})
    perplexity = np.exp(running_loss / counter)
    logger.log({f"{prefix}test_perplexity": perplexity})
    logger.log({f"{prefix}test_duration": duration})
    prefix = "Student" if is_student else "Teacher"
    print(f"{prefix} Test Loss: {(running_loss / counter):.5f} | Test Perplexity: {perplexity:.5f} | Duration: {duration:.5f} seconds")
    print(f"Ignored {ignored_count} batches")
    

def smart_to(model, device="cuda" if torch.cuda.is_available() else "mps"):
    if accelerator is None:
        return model.to(device)
    return model


def init_accelerate():
    global accelerator
    accelerator = Accelerator()
    print(f"Using device: {accelerator.device}")


def init_logger(logger_):
    global logger
    logger = logger_



# command line run for training with parsing arguments
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--load_chkpt", action="store_true")
    parser.add_argument("--load_hf_model", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--is_mamba", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--use_modified_tokenizer", action="store_true", default=False)
    parser.add_argument("--use_teacher_tokenizer", action="store_true", default=False)
    parser.add_argument("--teacher_model_path", type=str, default=teacher_model_path)
    parser.add_argument("--wandb_name", type=str, default='')
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--use_accelerate", action="store_true", default=False)
    parser.add_argument("--minimize_dataset", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--finetune_teacher", action="store_true", default=False)
    parser.add_argument("--hf_trainer", action="store_true", default=False)
    parser.add_argument("--optimizer", type=str, default="adamw_hf")
    parser.add_argument("--mixed_precision", action="store_true", default=False)
    parser.add_argument("--tf32", action="store_true", default=False)
    args = parser.parse_args()
    

    unique_run_id = str(random.randint(0, 1000)) + str(int(time.time()))[:4]
    name_prefix = f"{unique_run_id}_{args.wandb_name}_"
    log_config_dict = {
                "limit": str(args.limit),
                "batch_size": str(args.batch_size),
                "max_length": str(args.max_length),
                "epochs": str(args.epochs),
                "learning_rate": str(args.learning_rate),
                "model_path": str(args.model_path),
                "is_mamba": str(args.is_mamba),
                "accumulation_steps": str(args.accumulation_steps),
                "mixed_precision": str(args.mixed_precision),
                "tf32": str(args.tf32),
                
        }
    if args.use_accelerate:
        init_accelerate()
        accelerator.print("-----Accelerate Initialized-----")
        accelerator.init_trackers(
            project_name="HF-ACC",
            config=log_config_dict,
            init_kwargs={"wandb": {"name": f"mp{args.mixed_precision}-{args.epochs}-epochs-{args.max_length}-maxLen-alfa{args.alpha}-tmp{args.temperature}-{args.batch_size}-batchsize-{args.learning_rate}-lr-{args.is_mamba}-isMamba-{args.accumulation_steps}-accum-steps-{teacher_model_path}-teacher-model-{args.model_path}-student-model".replace('.','').replace('/','')}},
        )
        init_logger(accelerator)
    else:
        wandb.init(
            project="MMB-SE-KD-ULD",
            config=log_config_dict,
            name=None if args.resume else f"mp{args.mixed_precision}-{args.epochs}-epochs-{args.max_length}-maxLen-alfa{args.alpha}-tmp{args.temperature}-{args.batch_size}-batchsize-{args.learning_rate}-lr-{args.is_mamba}-isMamba-{args.accumulation_steps}-accum-steps-{teacher_model_path}-teacher-{args.model_path}-student".replace('.','').replace ('/',''),
            resume=args.resume,
            id=args.wandb_id
        )
        init_logger(wandb)

    if args.finetune_teacher:
        finetune_teacher(unique_id=name_prefix, batch_size=args.batch_size, max_length=args.max_length, minimize_dataset=args.minimize_dataset, epochs=args.epochs, lr=args.learning_rate, optimizer=args.optimizer, teacher_model_path=args.teacher_model_path, mixed_precision=args.mixed_precision, tf32=args.tf32)
    else:
        train(limit=args.limit, batch_size=args.batch_size, max_length=args.max_length, epochs=args.epochs,
            learning_rate=args.learning_rate, load_chkpt=args.load_chkpt, load_hf_model=args.load_hf_model,
            model_path=args.model_path, is_mamba=args.is_mamba, gpu=args.gpu, accumulation_steps=args.accumulation_steps,
            use_modified_tokenizer=args.use_modified_tokenizer, use_teacher_tokenizer=args.use_teacher_tokenizer,
            teacher_model_path=teacher_model_path, minimize_dataset=args.minimize_dataset, unique_id=name_prefix, alpha=args.alpha,
              temperature=args.temperature, hf_trainer=args.hf_trainer, optimizer=args.optimizer, mixed_precision=args.mixed_precision, tf32=args.tf32)

    # example command line run:
    # python evals/distillation.py --limit 1000000000000 --batch_size 8 --max_length 128 --epochs 3 --learning_rate 1e-3 --load_hf_model --model_path /cs/labs/roys/w552295/bamba/full_trained_epoch_2_lr_0.001_is_mamba_True_max_length_128  --is_mamba
    # python evals/distillation.py --limit 100 --batch_size 2 --max_length 128 --epochs 3 --learning_rate 1e-3 --load_hf_model --model_path state-spaces/mamba-370m-hf --accumulation_steps 16 --is_mamba



