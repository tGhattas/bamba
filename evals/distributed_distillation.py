import os
from typing import Union
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from itertools import islice
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from tqdm import tqdm
import numpy as np
import argparse
import wandb

def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
    if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:        
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "58055"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    return rank, world_size

rank, world_size = setup_distributed()
device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

teacher_model_path = "meta-llama/Meta-Llama-3-8B"
sanity_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def get_teacher_model(path: str):
    return AutoModelForCausalLM.from_pretrained(path)

def get_sanity_student_model(path: str = None):
    if path:
        model = AutoModelForCausalLM.from_pretrained(path)
    else:
        config = AutoConfig.from_pretrained(sanity_model_path)
        teacher_model_config = AutoConfig.from_pretrained(teacher_model_path)
        config.eos_token_id = teacher_model_config.eos_token_id
        config.bos_token_id = teacher_model_config.bos_token_id
        config.vocab_size = teacher_model_config.vocab_size
        config.num_hidden_layers = int(0.2 * config.num_hidden_layers)
        model = AutoModelForCausalLM.from_config(config)
    
    print_model_parameters("Sanity Student", model)
    return model

def get_mamba_model(path: str = None):
    teacher_model = get_teacher_model(teacher_model_path)
    param = next(teacher_model.parameters())
    teacher_dtype = param.dtype
    if path:
        return MambaLMHeadModel.from_pretrained(path, device=device, dtype=teacher_dtype)
    config_data = {
        "d_model": 2560,
        "n_layer": teacher_model.config.num_hidden_layers // 4,
        "vocab_size": teacher_model.config.vocab_size,
        "ssm_cfg": {},
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "pad_vocab_size_multiple": 8
    }
    config = MambaConfig(**config_data)
    
    mamba_student_model = MambaLMHeadModel(config,
                                           initializer_cfg=None,
                                           device=device,
                                           dtype=teacher_dtype)
    print(f"get_mamba_model: Number of hidden layers in the student model: {config.n_layer}")
    print_model_parameters("MAMBA", mamba_student_model)
    return mamba_student_model

temperature = 2.0
alpha = 0.5
HF_PADDING_IGNORE = -100

def init_dataloader(batch_size: int, max_length: int, partition: str = "train"):
    dataset_path = "wikitext-2-v1"
    dataset = load_dataset("wikitext", dataset_path)
    
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, use_fast=True)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    def tokenize_function(examples):
        return teacher_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=teacher_tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    data_loader = DataLoader(tokenized_datasets[partition], batch_size=batch_size, collate_fn=data_collator)
    return data_loader, teacher_tokenizer.pad_token_id

def print_model_parameters(model_name: str, model: Union[AutoModelForCausalLM, MambaLMHeadModel]):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}")
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Total Memory Footprint: {total_params * 4 / 1024 / 1024} MB")

def logits_to_tokens(logits):
    return torch.argmax(logits, dim=-1)

def distill_knowledge(teacher_model: AutoModelForCausalLM, student_model: Union[MambaLMHeadModel, AutoModelForCausalLM], optimizer: torch.optim.Optimizer,
                      batch_size: int, max_length: int, limit: int = 1000, epochs: int = 5, load_chkpt: bool = False, model_path: str = None, accumulation_steps: int = 1):
    if load_chkpt:
        student_model.load_state_dict(torch.load(model_path))

    first_batch = True
    log_interval = 200
    
    running_loss = 0
    running_distillation_loss = 0
    running_cross_entropy_loss = 0

    train_dataloader, pad_token_id = init_dataloader(batch_size, max_length, "train")
    eval_dataloader, _ = init_dataloader(batch_size, max_length, "test")
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=10, num_training_steps=epochs * len(train_dataloader))

    for epoch in range(epochs):
        for batch_idx, batch in tqdm(enumerate(islice(train_dataloader, limit))):
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

            distillation_loss = nn.KLDivLoss(reduction="batchmean")(
                torch.log_softmax(student_outputs / temperature, dim=-1),
                torch.softmax(teacher_outputs / temperature, dim=-1),
            ) * (temperature ** 2)

            student_label_loss = nn.CrossEntropyLoss(ignore_index=HF_PADDING_IGNORE)(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))
            
            loss = alpha * distillation_loss + (1 - alpha) * student_label_loss
            running_loss += loss.item()
            running_distillation_loss += distillation_loss.item()
            running_cross_entropy_loss += student_label_loss.item()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.zero_grad()
                loss.backward()
                lr_scheduler.step()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=0.5)
                optimizer.step()

            if batch_idx % log_interval == 0:
                wandb.log({"epoch": epoch, "running_loss": running_loss / log_interval})
                wandb.log({"epoch": epoch, "running_distillation_loss": running_distillation_loss / log_interval})
                wandb.log({"epoch": epoch, "running_cross_entropy_loss": running_cross_entropy_loss / log_interval})
                running_loss = 0
                running_distillation_loss = 0
                running_cross_entropy_loss = 0

            if batch_idx % (log_interval * 4) == 0:
                evaluate(student_model)
                
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")
        torch.save(student_model.state_dict(), f"./checkpoints/distr_student_chkpt_epoch_{epoch}_type_{'mamba' if isinstance(student_model, MambaLMHeadModel) else 'transformer'}_max_length_{max_length}.pt")

def train(limit: int = 1000, batch_size: int = 4, max_length: int = 128, epochs: int = 5,
          learning_rate: float = 5e-5, load_chkpt: bool = False, load_hf_model: bool = False, model_path: str = None, is_mamba: bool = False, accumulation_steps: int = 1):
       # ensure that the distributed environment is properly set up
    rank, _ = setup_distributed()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Initialize the teacher model
    teacher_model = get_teacher_model(teacher_model_path)
    teacher_model.to(device)
    teacher_model = DDP(teacher_model, device_ids=[rank])
    teacher_model.eval()
    print_model_parameters(teacher_model_path, teacher_model)

    # Initialize the student model
    if load_hf_model:
        if not is_mamba:
            student_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        else:
            student_model = get_mamba_model(path=model_path)
    else:
        if not is_mamba:
            student_model = get_sanity_student_model().to(device)
        else:
            student_model = get_mamba_model()

    student_model.to(device)
    student_model = DDP(student_model, device_ids=[rank])
    student_model.train()
    
    # Set up the optimizer
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    
    # Perform knowledge distillation
    distill_knowledge(teacher_model, student_model, optimizer, batch_size, max_length, limit=limit, epochs=epochs,
                      load_chkpt=load_chkpt, model_path=model_path, accumulation_steps=accumulation_steps)
    
    # Save the final student model
    if rank == 0:
        student_model.module.save_pretrained(f"full_trained_epoch_{epochs}_lr_{learning_rate}_is_mamba_{is_mamba}_max_length_{max_length}")

def evaluate(model_or_path: Union[str, AutoModelForCausalLM, MambaLMHeadModel], eval_dataloader: DataLoader = None, pad_token_id: int = None):
    if isinstance(model_or_path, str):
        student_model = AutoModelForCausalLM.from_pretrained(model_or_path)
    else:
        student_model = model_or_path
    
    student_model.eval()
    rank, _ = setup_distributed()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    student_model.to(device)

    if eval_dataloader is None:
        dataloader, pad_token_id = init_dataloader(8, 128, "test")
    else:
        dataloader, pad_token_id = eval_dataloader, pad_token_id

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
    parser.add_argument("--accumulation_steps", type=int, default=1)

    args = parser.parse_args()

    wandb.init(
        project="MAMBA-KD-dist",
        config={
            "limit": args.limit,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "model_path": args.model_path,
            "is_mamba": args.is_mamba,
            "accumulation_steps": args.accumulation_steps
        }
    )

    train(limit=args.limit, batch_size=args.batch_size, max_length=args.max_length, epochs=args.epochs,
          learning_rate=args.learning_rate, load_chkpt=args.load_chkpt, load_hf_model=args.load_hf_model,
          model_path=args.model_path, is_mamba=args.is_mamba, accumulation_steps=args.accumulation_steps)