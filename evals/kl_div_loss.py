

from torch import nn
import torch

class KLDivLoss(nn.Module):

    def __init__(self, reduction='mean', temperature=1.0, ignore_idx=-100, distillation_loss_weight=0.5, using_acc=False, scaling_factor=None, *args, **kwargs):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.ignore_idx = ignore_idx
        self.distillation_loss_weight = distillation_loss_weight
        self.using_acc = using_acc
        self.scaling_factor = scaling_factor

    def forward(self, student_outputs, teacher_outputs, labels, *args, **kwargs):
        
        student_outputs = student_outputs.logits
        teacher_outputs = teacher_outputs.logits
        if not self.using_acc:
            device = "cuda" if torch.cuda.is_available() else "mps"
            student_outputs = student_outputs.to(device)
            teacher_outputs = teacher_outputs.to(device)
        assert student_outputs.shape == teacher_outputs.shape, f"Student logits shape: {student_outputs.shape} != Teacher logits shape: {teacher_outputs.shape}"
        if student_outputs.dtype != teacher_outputs.dtype:
            student_outputs = student_outputs.to(teacher_outputs.dtype)
        # Compute the distillation loss based on https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
        if self.distillation_loss_weight == 0:
            loss = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))
            return loss, loss, torch.tensor(0, dtype=loss.dtype, device=loss.device)
        
        distillation_loss = nn.KLDivLoss(reduction="batchmean")(
            torch.log_softmax(student_outputs / self.temperature, dim=-1),
            torch.softmax(teacher_outputs / self.temperature, dim=-1),
        ) * (self.temperature ** 2)

        scaling_factor = (100 if self.temperature != 1 else 10) if self.scaling_factor is None else self.scaling_factor
        student_label_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))
        loss = self.distillation_loss_weight * distillation_loss / scaling_factor + (1 - self.distillation_loss_weight) * student_label_loss
            
        return loss, student_label_loss, distillation_loss