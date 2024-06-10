import os
from typing import Union
import torch
import torch.nn as nn
from torch.nn import DataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
from itertools import islice
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from tqdm import tqdm
import numpy as np
import argparse
import time
# WANDB
import wandb




teacher_model_path = "meta-llama/Meta-Llama-3-8B"
# teacher_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# teacher_model_path = "mistralai/Mistral-7B-v0.3"
def get_teacher_model(path: str):
    return AutoModelForCausalLM.from_pretrained(path)


def get_sanity_student_model(path: str):
    model = AutoModelForCausalLM.from_pretrained(path) 
    model.num_hidden_layers = model.num_hidden_layers // 4
    return model



# MAMBA student model
def get_mamba_model(path: str = None, gpu: int = None):
    device = f'cuda{f":{gpu}" if gpu else ""}'
    teacher_model = get_teacher_model(teacher_model_path)
    
    if path:
         return MambaLMHeadModel.from_pretrained(path, device=device, dtype=teacher_dtype)
    config_data = {
        "d_model": 2560,
        "n_layer": teacher_model.config.num_hidden_layers // 5, # 22 in case of TinyLlama-1.1B
        "vocab_size": teacher_model.config.vocab_size,
        "ssm_cfg": {},
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "pad_vocab_size_multiple": 8
    }
    config = MambaConfig(**config_data)
    param = next(teacher_model.parameters())
    teacher_dtype = param.dtype
    mamba_student_model = MambaLMHeadModel(config,
            initializer_cfg=None,
            device=device,
            dtype=teacher_dtype,
            )
    return mamba_student_model



# Step 3: Knowledge Distillation

temperature = 2.0  # Temperature for softmax computation
alpha = 0.5  # The weight of the distillation loss


HF_PADDING_IGNORE = -100



def init_dataloader(batch_size: int, max_length: int, partition: str = "train"):

    dataset_path = "wikitext-2-v1"
    dataset = load_dataset("wikitext", dataset_path, streaming=True) #

    
    # Load the teacher tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, use_fast=True)

    #add paddign token to the tokenizer
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Tokenize the dataset
    def tokenize_function(examples):
        return teacher_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=teacher_tokenizer,
        mlm=False,  # Set to True if using Masked Language Modeling
        pad_to_multiple_of=8  # Optional, can pad to the nearest multiple of 8 for efficiency
    )
    
    # Create the data loader
    data_loader = DataLoader(tokenized_datasets[partition], batch_size=batch_size, collate_fn=data_collator)
    return data_loader, teacher_tokenizer.pad_token_id


def print_model_parameters(model_name: str, model: Union[AutoModelForCausalLM, MambaLMHeadModel]):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}")
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

def logits_to_tokens(logits):
    """Convert logits to token ids."""
    return torch.argmax(logits, dim=-1)

def distill_knowledge(teacher_model: AutoModelForCausalLM, student_model: Union[MambaLMHeadModel, AutoModelForCausalLM], optimizer: torch.optim.Optimizer,
                       batch_size: int, max_length: int, limit: int=1000, epochs: int=5, load_chkpt: bool=False, model_path: str=None, gpu: int = None):
    device = f'cuda{f":{gpu}" if gpu else ""}'
    if load_chkpt:
        student_model.load_state_dict(torch.load(model_path))

    first_batch = True
    log_interval = 200
    
    # print the number of parameters in both models
    print_model_parameters(teacher_model_path, teacher_model)
    print_model_parameters(f"{'MAMBA' if isinstance(student_model, MambaLMHeadModel) else 'Transformer'} model", student_model)

    running_loss = 0
    running_distillation_loss = 0
    running_cross_entropy_loss = 0

    for epoch in range(epochs):

        
        dataloader, pad_token_id = init_dataloader(batch_size, max_length)
        for batch_idx, batch in tqdm(enumerate(islice(dataloader, limit))):
            batched_input_ids = batch['input_ids'].to(device)
            
            inputs = batched_input_ids[:, :-1].contiguous().to(device)
            labels = batched_input_ids[:, 1:].contiguous().to(device)
            labels[labels == pad_token_id] = HF_PADDING_IGNORE

            batched_attention_mask = batch['attention_mask'].to(device)
            attention_mask = batched_attention_mask[:, :-1].contiguous().to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=inputs,
                                                 attention_mask=attention_mask
                                                ).logits.to(device)
            if isinstance(student_model, MambaLMHeadModel):
                student_outputs = student_model(input_ids=inputs,
                                                ).logits.to(device)
            else:
                student_outputs = student_model(input_ids=inputs,
                                                attention_mask=attention_mask
                                                ).logits.to(device) 

            if first_batch:
                print(f"Student logits shape: {student_outputs.shape}")
                print(f"Teacher logits shape: {teacher_outputs.shape}")
                first_batch = False

            # Compute the distillation loss based on https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
            distillation_loss = nn.KLDivLoss(reduction="batchmean")(
                torch.log_softmax(student_outputs / temperature, dim=-1),
                torch.softmax(teacher_outputs / temperature, dim=-1),
            ) * (temperature ** 2)

            student_label_loss = nn.CrossEntropyLoss(ignore_index=HF_PADDING_IGNORE)(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))
            
            loss = alpha * distillation_loss + (1 - alpha) * student_label_loss
            running_loss += loss.item()
            running_distillation_loss += distillation_loss.item()
            running_cross_entropy_loss += student_label_loss.item()

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=0.5)
            
            optimizer.step()

            if batch_idx % log_interval == 0:
                wandb.log({"epoch": epoch, "runningl_loss": running_loss / log_interval})
                wandb.log({"epoch": epoch, "running_distillation_loss": running_distillation_loss / log_interval})
                wandb.log({"epoch": epoch, "runnning_cross_entropy_loss": running_cross_entropy_loss / log_interval})
                running_loss = 0
                running_distillation_loss = 0
                running_cross_entropy_loss = 0
                
                
                
        if os.path.exists("./checkpoints") == False:
            os.mkdir("./checkpoints")
        torch.save(student_model.state_dict(), f"./checkpoints/student_chkpt_epoch_{epoch}.pt")

        # evaluate the student model
        evaluate(student_model, gpu=gpu)


# Training Loop
def train(limit: int = 1000, batch_size: int = 4, max_length: int = 128, epochs: int = 5,
           learning_rate: float = 5e-5, load_chkpt: bool=False, load_hf_model: bool=False, model_path: str=None, is_mamba: bool=False, gpu: int = None):   
    # assert that if either load_chkpt or load_hf_model is True but not both
    assert not (load_chkpt and load_hf_model), "Both load_chkpt and load_hf_model cannot be True at the same time"
    device = f'cuda{f":{gpu}" if gpu else ""}'
    teacher_model = get_teacher_model(teacher_model_path)
    teacher_model.to(device)
    teacher_model = DataParallel(teacher_model)
    teacher_model.eval()
    if load_hf_model:
        if not is_mamba:
            student_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        else:
            student_model = get_mamba_model(path=model_path, gpu=gpu)
    else:
        if not is_mamba:
            student_model = get_sanity_student_model(teacher_model_path).to(device)
        else:
            student_model = get_mamba_model(gpu=gpu)
    
    student_model.train()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    distill_knowledge(teacher_model, student_model, optimizer, batch_size, max_length, limit=limit, epochs=epochs,
                       load_chkpt=load_chkpt, model_path=model_path, gpu=gpu)
    # save the student model 
    student_model.save_pretrained(f"full_trained_epoch_{epochs}_lr_{learning_rate}_is_mamba_{is_mamba}_max_length_{max_length}")


# Evaluate the student model
def evaluate(model_or_path: Union[str, AutoModelForCausalLM, MambaLMHeadModel], gpu: int = None):

    # evaluate the student model using the test dataset
    if isinstance(model_or_path, str):
        student_model = AutoModelForCausalLM.from_pretrained(model_or_path).to(f'cuda:{gpu}')
    else:
        student_model = model_or_path
    
    student_model.eval()
    dataloader, pad_token_id = init_dataloader(4, 256, "test")
    device = f'cuda{f":{gpu}" if gpu else ""}'
    # evalua using the test dataset
    running_loss = 0
    counter = 0
    for batch in tqdm(dataloader):
        counter += 1
        batched_input_ids = batch['input_ids'].to(device)
        inputs = batched_input_ids[:, :-1].contiguous().to(device)
        labels = batched_input_ids[:, 1:].contiguous().to(device)
        labels[labels == pad_token_id] = HF_PADDING_IGNORE

        batched_attention_mask = batch['attention_mask'].to(device)
        attention_mask = batched_attention_mask[:, :-1].contiguous().to(device)

        if isinstance(student_model, MambaLMHeadModel):
            student_outputs = student_model(input_ids=inputs,
                                        ).logits.to(device)
        else:
            student_outputs = student_model(input_ids=inputs,
                                            attention_mask=attention_mask
                                            ).logits.to(device)

        student_label_loss = nn.CrossEntropyLoss(ignore_index=HF_PADDING_IGNORE)(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))
        running_loss += student_label_loss.item()

            
    wandb.log({"test_loss": running_loss / counter})
    perplexity = np.exp(running_loss / counter)
    wandb.log({"test_perplexity": perplexity})
    

    
    

        

# command line run for training with parsing arguments
if __name__ == "__main__":
    
    # train(limit=1000000000, batch_size=8, max_length=256, epochs=5, learning_rate=1e-4, is_mamba=True, gpu=0)
    

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
    args = parser.parse_args()

    wandb.init(
        project="MAMBA-KD",
        config={
                "limit": str(args.limit),
                "batch_size": str(args.batch_size),
                "max_length": str(args.max_length),
                "epochs": str(args.epochs),
                "learning_rate": str(args.learning_rate),
                "model_path": str(args.model_path),
                "is_mamba": str(args.is_mamba),
        }
       )

    train(limit=args.limit, batch_size=args.batch_size, max_length=args.max_length, epochs=args.epochs,
          learning_rate=args.learning_rate, load_chkpt=args.load_chkpt, load_hf_model=args.load_hf_model,
          model_path=args.model_path, is_mamba=args.is_mamba, gpu=args.gpu)

    # example command line run:
    # python evals/distillation.py --limit 1000000000000 --batch_size 16 --max_length 256 --epochs 5 --learning_rate 1e-4 --is_mamba --gpu 0
