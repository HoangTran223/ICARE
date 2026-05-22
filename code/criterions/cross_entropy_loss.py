import torch
import torch.nn as nn
import torch.distributed as dist


class CrossEntropyLoss(nn.Module):
    def __init__(self, args, padding_id=-100) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.label_smoothing = args.label_smoothing
        self.padding_id = padding_id
    
    def forward(self, distiller, batch, logging_output):
        self.distiller = distiller
        model = distiller.student_model
        logits = model(**batch["input_batch"]).logits
        loss, nll_loss = self.compute_cross_entropy_loss(logits, batch["label_batch"]["label"])
        # print(loss)
        # print(nll_loss)
        accuracy = self.compute_token_accuracy(
            logits, 
            batch["label_batch"], 
        )
        logging_output = self.record_logging_output(
            logging_output, 
            {
                "loss": loss / batch["label_batch"]["loss_denom"],
                "nll_loss": nll_loss / batch["label_batch"]["loss_denom"],
                "accuracy": accuracy
            }
        )
        return loss / batch["label_batch"]["loss_denom"], logging_output

    def compute_cross_entropy_loss(self, logits, target, coef=None, reduction="sum", log=None):
        # target = label_batch["label"]
        pad_mask = target.ne(self.padding_id)
        target = target.unsqueeze(-1)
        target = torch.where(
            target.eq(-100), 
            torch.zeros_like(target),
            target
        )
        logits = logits.masked_fill_(logits.isnan() | logits.isinf(), 0.0)
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        nll_loss = -lprobs.gather(-1, target).squeeze(-1)
        
        if reduction == "sum":
            if coef is not None:
                nll_loss = (nll_loss * coef * pad_mask).sum()
            else:
                nll_loss = (nll_loss * pad_mask).sum()
        elif reduction == "mean":
            nll_loss = (nll_loss * pad_mask).sum() / pad_mask.sum()
        
        if self.label_smoothing > 0:
            eps_i = self.label_smoothing / (lprobs.shape[-1] - 1)
            smooth_loss = -lprobs.sum(-1)
            loss = (1 - self.label_smoothing - eps_i) * nll_loss + eps_i * smooth_loss
            if reduction == "sum":
                loss = (loss * pad_mask).sum()
            elif reduction == "mean":
                loss = (loss * pad_mask).sum() / pad_mask.sum()
        else:
            loss = nll_loss
        
        if log is not None:
            log["nll_loss"] = nll_loss
        
        return loss, nll_loss

    def compute_token_accuracy(self, logits, labels, loss_denom=None):
        """Return token accuracy or correct-token count.

        - ``labels`` is a dict with ``label`` and ``loss_denom`` (legacy batch API).
        - ``labels`` is a label tensor (``distillation_baseline`` API): returns the
          count of correct tokens; divide by ``batch_denom`` in ``record_logging_output``.
        """
        if isinstance(labels, dict):
            target = labels["label"]
            denom = labels["loss_denom"]
        else:
            target = labels
            denom = loss_denom
        pad_mask = target.ne(self.padding_id)
        correct = (logits.argmax(-1).eq(target) * pad_mask).sum()
        if isinstance(labels, dict):
            return correct / denom
        if loss_denom is not None:
            return correct / loss_denom
        return correct
    
    def record_logits(self, logits, target, log, teacher_logits=None, teacher_target=None):
        pad_mask = target.eq(self.padding_id)
        pos_target = torch.where(
            pad_mask,
            torch.zeros_like(target),
            target
        )
        target_logits = logits.gather(-1, pos_target.unsqueeze(-1)).squeeze(-1)
        non_target_logits = (logits.sum(-1) - target_logits) / (logits.shape[-1] - 1)
        target_logits = target_logits.masked_fill_(pad_mask, 0.0).sum()
        non_target_logits = non_target_logits.masked_fill_(pad_mask, 0.0).sum()
        
        log["target_logits"] = target_logits
        log["non_target_logits"] = non_target_logits

        if teacher_logits is not None:
            assert teacher_target is not None
            teacher_pad_mask = teacher_target.eq(self.padding_id)
            pos_teacher_target = torch.where(
                teacher_pad_mask,
                torch.zeros_like(teacher_target),
                teacher_target
            )
            teacher_target_logits = teacher_logits.gather(-1, pos_teacher_target.unsqueeze(-1)).squeeze(-1)
            teacher_non_target_logits = (teacher_logits.sum(-1) - teacher_target_logits) / (teacher_logits.shape[-1] - 1)
            teacher_target_logits = teacher_target_logits.masked_fill_(teacher_pad_mask, 0.0).sum()
            teacher_non_target_logits = teacher_non_target_logits.masked_fill_(teacher_pad_mask, 0.0).sum()
            
            log["teacher_target_logits"] = teacher_target_logits
            log["teacher_non_target_logits"] = teacher_non_target_logits
    
    def record_logging_output(self, logging_output, batch_denom_or_content, log=None):
        """Aggregate metrics across ranks.

        Two call styles:
        - ``record_logging_output(logging_output, content_dict)`` — values already scaled.
        - ``record_logging_output(logging_output, batch_denom, log_dict)`` — raw sums
          divided by ``batch_denom`` (used by DSKD / SRA / IMPACT criterions).
        """
        if log is None:
            content = batch_denom_or_content
            denom = 1.0
        else:
            content = log
            denom = batch_denom_or_content
            if torch.is_tensor(denom):
                denom = denom.item()
            else:
                denom = float(denom)

        world = dist.get_world_size()
        for k, v in content.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(float(v), dtype=torch.float32)
            dist.all_reduce(v, dist.ReduceOp.SUM)
            val = v.item() / (world * denom)
            if k in logging_output:
                logging_output[k].append(val)
            else:
                logging_output[k] = [val]
        return logging_output