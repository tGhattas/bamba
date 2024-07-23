from trl import SFTTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from kl_div_loss import KLDivLoss


class KDTrainer(SFTTrainer):

    def __init__(self, teacher_model=None, student_model=None, temperature=None, alfa=None, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alfa = alfa
        self.kd_loss = KLDivLoss(temperature=temperature, distillation_loss_weight=alfa)
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            print("-------------------Using accelerator in KDTrainer-------------------")
            self.teacher_model = self.accelerator.prepare(teacher_model)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.teacher_model.to(device)
        self.teacher_model.eval()
    
    def compute_loss(self, model, inputs, return_outputs=False):
        student_outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        labels = inputs.get("labels")
        loss, student_label_loss, distillation_loss = self.kd_loss(student_outputs, teacher_outputs, labels)
        self.log({"student_label_loss": student_label_loss, "distillation_loss": distillation_loss})
        return (loss, student_outputs) if return_outputs else loss
