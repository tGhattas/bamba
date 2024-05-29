from typing import Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from itertools import islice
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from tqdm import tqdm

# WANDB
import wandb
wandb.init()
wandb_outputs_table = wandb.Table(columns=["input_text", "label_text", "student_output_text", "teacher_output_text"])

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\033[93m\033[1mDevice is: {device}\033[0m")

# Step 1: Load the teacher model (Mistral 7B as a LMHeadModel)
teacher_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path).to(device)
teacher_model.eval()


# Step 2: Define the student model (a smaller transformer/MAMBA model with a LM head)

# for sanity check, here is a TinyLlama student model that is identical to teacher but without the pre-trained weights and half the hidden size
sanity_student_config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
sanity_student_config.num_hidden_layers = sanity_student_config.num_hidden_layers // 2
llama_student_model = AutoModelForCausalLM.from_config(sanity_student_config).to(device)

# MAMBA student model
config_data = {
    "d_model": 2560,
    "n_layer": teacher_model.config.num_hidden_layers, # 22 in case of TinyLlama-1.1B
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

vocab_size = teacher_model.config.vocab_size

student_model = llama_student_model


# Step 3: Knowledge Distillation

temperature = 2.0  # Temperature for softmax computation
alpha = 0.7  # The weight of the distillation loss


HF_PADDING_IGNORE = -100



def init_dataloader(batch_size: int = 4):
    # dataset_path = "monology/pile-uncopyrighted"
    # dataset = load_dataset(dataset_path, streaming=True)
    dataset_path = "wikitext-2-v1"
    dataset = load_dataset("wikitext", dataset_path)

    
    # Load the teacher tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)

    # Tokenize the dataset
    def tokenize_function(examples):
        return teacher_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256, return_tensors="pt")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Create the data loader
    data_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, num_workers=4)
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

def distill_knowledge(teacher_model: AutoModelForCausalLM, student_model: MambaLMHeadModel, dataloader: DataLoader,
                       optimizer: torch.optim.Optimizer, pad_token_id: int, limit: int=1000):
    # TODO remove
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    first_batch = True
    log_interval = 50
    epochs = 1
    # print the number of parameters in both models
    print_model_parameters(teacher_model_path, teacher_model)
    print_model_parameters("MAMBA Student Model", student_model)
    for epoch in range(epochs):
        for batch_idx, batch in tqdm(enumerate(islice(dataloader, limit))):
            batched_input_ids = batch['input_ids'].to(device)
            
            
            inputs = batched_input_ids[:, :-1].contiguous().to(device)
            labels = batched_input_ids[:, 1:].contiguous().to(device)
            # labels[labels == pad_token_id] = HF_PADDING_IGNORE

            batched_attention_mask = batch['attention_mask'].to(device)
            attention_mask = batched_attention_mask[:, :-1].contiguous().to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=inputs,
                                                 attention_mask=attention_mask
                                                ).logits.to(device)
            
            student_outputs = student_model(input_ids=inputs,
                                            attention_mask=attention_mask
                                             ).logits.to(device) # TODO pass the attention mask to mamba also

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
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

            optimizer.step()

            if batch_idx % log_interval == 0:
                # Decode and log the input, label, and model output
                decoded_inputs = teacher_tokenizer.batch_decode(inputs)
                decoded_labels = teacher_tokenizer.batch_decode(labels)
                decoded_student_outputs = teacher_tokenizer.batch_decode(logits_to_tokens(student_outputs))
                decoded_teacher_outputs = teacher_tokenizer.batch_decode(logits_to_tokens(teacher_outputs))
                for i in range(len(decoded_inputs)):
                    wandb_outputs_table.add_data(decoded_inputs[i], decoded_labels[i], decoded_student_outputs[i], decoded_teacher_outputs[i])
                wandb.log({"outputs": wandb_outputs_table})

                wandb.log({"epoch": epoch, "loss": loss.item()})
                wandb.log({"epoch": epoch, "distillation_loss": distillation_loss.item()})
                wandb.log({"epoch": epoch, "cross_entropy_loss": student_label_loss.item()})
                
            # report to wandb

# Step 4: Training Loop
def train(limit: int = 1000, batch_size: int = 4):        
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    dataloader, pad_token_id = init_dataloader(batch_size)
    teacher_model.eval()
    student_model.train()
    distill_knowledge(teacher_model, student_model, dataloader, optimizer, pad_token_id, limit=limit)
    # save the student model 
    student_model.save_pretrained("student_model")

# Step 5: Evaluate the student model
def evaluate(student_model_path: str):
    student_model = MambaLMHeadModel.from_pretrained(student_model_path).to(device)
    student_model.eval()
    dataloader = init_dataloader()
    for batch in tqdm(dataloader):
        batch = batch['input_ids'].to(device)
        inputs = batch[:, :-1].contiguous().to(device)
        labels = batch[:, 1:].contiguous().to(device)
        student_outputs = student_model(input_ids=inputs)
        student_label_loss = nn.CrossEntropyLoss()(student_outputs.logits.view(-1, student_outputs.logits.size(-1)), labels.view(-1))
        print(f"Student loss: {student_label_loss.item()}")
