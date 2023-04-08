import torch
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

class DistillationTrainer(Trainer):

    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits

        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        loss = loss_fct(
            torch.nn.functional.log_softmax(student_logits / self.args.temperature, dim=-1),
            torch.nn.functional.softmax(teacher_logits / self.args.temperature, dim=-1)
        )

        return (loss, student_outputs) if return_outputs else loss
