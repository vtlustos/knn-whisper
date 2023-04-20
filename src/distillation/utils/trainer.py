import torch
import tqdm
from transformers import (Seq2SeqTrainer)
class DistillationTrainer(Seq2SeqTrainer):

    def __init__(self, config, device, student_model, teacher_model, 
                 train_dataset, eval_dataset, tokenizer,
                 data_collator, compute_metrics, 
                 temperature=2.0, supervised=False):
        
        super().__init__(
            args=config,
            model=student_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        self.device=device
        self.student = student_model
        self.teacher = teacher_model
        self.temperature = temperature
        self.supervised = supervised
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean", 
                                          log_target=True)
        self.scaler = torch.cuda.amp.GradScaler()

        #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #self.train_loader = torch.utils.data.DataLoader(train_dataset, 
        #    batch_size=config.per_device_train_batch_size,
        #    sampler=train_sampler
        #)
    
    def compute_loss(self, inputs):

        labels = inputs.pop("labels")
        student_outputs = self.student(**inputs, use_cache=False)
        student_logits = student_outputs.get("logits")

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, use_cache=False)
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

        return loss
    
    def train(self):

        for epoch in tqdm(range(self.args.max_steps)):

            for step, batch in tqdm(enumerate(self.get_train_dataloader())):

                inputs = self._prepare_inputs(batch)
                inputs = {k: v.to(self.device) for k, v in inputs.items() \
                          if isinstance(v, torch.Tensor)}
                
                loss = self.compute_loss(inputs)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step()

                if self.args.local_rank in [-1, 0] and \
                        self.args.logging_steps > 0 and \
                        step % self.args.logging_steps == 0:
                    
                    logs = {}
                    logs["loss"] = loss.item()
                    self.log(logs)

                if self.args.local_rank in [-1, 0] and \
                    self.args.save_steps > 0 and \
                    step % self.args.save_steps == 0:

                    self.save_model()

                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    return

            self.evaluate()








