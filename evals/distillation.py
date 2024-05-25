import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from itertools import islice
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from tqdm import tqdm
import wandb
wandb.init()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\033[93m\033[1mDevice is: {device}\033[0m")

# Step 1: Load the teacher model (Mistral 7B as a LMHeadModel)
teacher_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path).to(device)
teacher_model.eval()

# Step 2: Define the student model (a smaller transformer model with a LM head)
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
student_model = MambaLMHeadModel(config,
        initializer_cfg=None,
        device=device,
        dtype=teacher_dtype,
        )


# Step 3: Knowledge Distillation

temperature = 2.0  # Temperature for softmax computation
alpha = 0.5  # The weight of the distillation loss

def init_dataloader():
    dataset = load_dataset("monology/pile-uncopyrighted", streaming=True)

    # Load the teacher tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)

    # Tokenize the dataset
    def tokenize_function(examples):
        return teacher_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Create the data loader
    data_loader = DataLoader(tokenized_datasets["train"], batch_size=8, num_workers=4)
    return data_loader


def distill_knowledge(teacher_model: AutoModelForCausalLM, student_model: MambaLMHeadModel, dataloader: DataLoader, optimizer: torch.optim.Optimizer, limit=1000):
    student_model.train()
    first_batch = True
    log_interval = 10
    epochs = 1
    for epoch in range(epochs):
        for batch_idx, batch in tqdm(enumerate(islice(dataloader, limit))):
            batch = batch['input_ids'].to(device)
            inputs = batch[:, :-1].contiguous().to(device)
            labels = batch[:, 1:].contiguous().to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=inputs).logits.to(device)
            
            student_outputs = student_model(input_ids=inputs)

            if first_batch:
                print(f"Student logits shape: {student_outputs.logits.shape}")
                print(f"Teacher logits shape: {teacher_outputs.shape}")
                first_batch = False

            # Compute the distillation loss based on https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
            distillation_loss = nn.KLDivLoss(reduction="batchmean")(
                torch.log_softmax(student_outputs.logits / temperature, dim=-1),
                torch.softmax(teacher_outputs / temperature, dim=-1),
            ) * (temperature ** 2)

            student_label_loss = nn.CrossEntropyLoss()(student_outputs.logits.view(-1, student_outputs.logits.size(-1)), labels.view(-1))
            loss = alpha * distillation_loss + (1 - alpha) * student_label_loss
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                wandb.log({"epoch": epoch, "loss": loss.item()})
            # report to wandb

# Step 4: Training Loop
def train(limit: int = 1000):        
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    dataloader = init_dataloader()

    distill_knowledge(teacher_model, student_model, dataloader, optimizer, limit=limit)
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
