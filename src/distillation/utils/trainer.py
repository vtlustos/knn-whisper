import torch
from transformers import (Seq2SeqTrainer)

class DistillationTrainer(Seq2SeqTrainer):

    def __init__(self, teacher_model, temperature, supervised=False, *args, **kwargs):
        super().__init__(*args, **kwargs)       
        self.teacher = teacher_model
        self.temperature = temperature
        self.supervised = supervised
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        student_outputs = model(**inputs)
        student_logits = student_outputs.get("logits")

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.get("logits")

        kl_loss = self.kl_loss(
            torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1),
            torch.nn.functional.log_softmax(teacher_logits / self.temperature, dim=-1)
        )
        loss = kl_loss

        if self.supervised:
            alpha = 0.5
            ce_loss = self.ce_loss(student_logits, labels)
            loss = alpha * ce_loss + (1 - alpha) * kl_loss

        return (loss, student_outputs) if return_outputs else loss
