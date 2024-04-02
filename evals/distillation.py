import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from itertools import islice
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig

# Step 1: Load the teacher model (Mistral 7B as a LMHeadModel)
teacher_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path)
teacher_model.eval()

# Step 2: Define the student model (a smaller transformer model with a LM head)
config_data = {
    "d_model": 2560,
    "n_layer": teacher_model.config.num_hidden_layers, # 22 in case of TinyLlama-1.1B
    "vocab_size": 50277,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
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
    data_loader = DataLoader(tokenized_datasets["train"], batch_size=32)
    return data_loader


def distill_knowledge(teacher_model, student_model, dataloader, optimizer):
    student_model.train()
    limit = 1000
    for inputs, labels in islice(dataloader, limit):

        optimizer.zero_grad()
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs).logits
        
        student_outputs = student_model(**inputs)

        # Compute the distillation loss based on https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
        distillation_loss = nn.kl_div(
            torch.log_softmax(student_outputs / temperature, dim=-1),
            torch.softmax(teacher_outputs / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        student_label_loss = nn.cross_entropy(student_outputs, labels)
        loss = alpha * distillation_loss + (1 - alpha) * student_label_loss
        loss.backward()
        optimizer.step()

# Step 4: Training Loop
def train():        
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    dataloader = init_dataloader()
    distill_knowledge(teacher_model, student_model, dataloader, optimizer)

# Step 5: Evaluate the student model
# ...
