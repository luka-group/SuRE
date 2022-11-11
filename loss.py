import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from transformers import AutoTokenizer
from dataclasses import dataclass


class NALoss(object):
	def __init__(self, temp_file_path, tokenizer):
		self.temp_file_path = temp_file_path
		self.tokenizer = tokenizer
		self.init_temp_set()

	def init_temp_set(self):
		with open(self.temp_file_path) as f:
			templates = json.load(f)
		words = []
		for rel, template in templates.items():
			words += template.replace("{subj}", "").replace("{obj}", "").strip().split(" ")
		words = set(words)
		self.word_idx = []
		for word in words:
			self.word_idx += self.tokenizer(word, add_special_tokens=False)["input_ids"]
		self.word_idx = list(set(self.word_idx))

	def __call__(self, logits):
		logits = F.softmax(logits, dim=2)
		logits = logits[:, :, self.word_idx]
		loss = torch.max(torch.log(logits.view(-1) + 1e-10))
		return loss

@dataclass
class LabelSmoother:
	epsilon: float = 0.1
	fairseq: bool = False
	ignore_index: int = -100

	def __call__(self, model_output, labels):
		logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
		log_probs = -nn.functional.log_softmax(logits, dim=-1)
		if labels.dim() == log_probs.dim() - 1:
			labels = labels.unsqueeze(-1)

		padding_mask = labels.eq(self.ignore_index)
		# In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
		# will ignore them in any case.
		labels = torch.clamp(labels, min=0)
		nll_loss = log_probs.gather(dim=-1, index=labels)
		# works for fp16 input tensor too, by internally upcasting it to fp32
		smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

		nll_loss.masked_fill_(padding_mask, 0.0)
		smoothed_loss.masked_fill_(padding_mask, 0.0)

		num_active_elements = padding_mask.numel() - padding_mask.long().sum()

		# huggingface implementation
		# Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
		nll_loss = nll_loss.sum() / num_active_elements
		smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
		return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


if __name__ == "__main__":
	temp_file_path = "/nas/home/keminglu/dataset/semeval/rel2temp.json"
	tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
	na_loss_fct = NALoss(temp_file_path, tokenizer)
	logits = torch.ones(8, 100, 70000)
	print(na_loss_fct(logits))
