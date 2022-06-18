import json
import copy
import torch
from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration

class Trie(object):
	def __init__(self, templates, tokenizer, type_constraint):
		self.templates = templates
		self.tokenizer = tokenizer
		self.type_constraint = type_constraint

	def build_tree(self, subj, obj, subj_type, obj_type):
		if self.type_constraint is not None:
			rels = self.type_constraint[f"{subj_type}|{obj_type}"]
			all_templates = [self.templates[rel] for rel in rels]
			all_templates.append(self.templates["no_relation"])
		else:
			all_templates = list(self.templates.values())
		tree = {2:{0:{}}}
		for temp in all_templates:
			temp = temp.format(subj=subj, obj=obj)
			temp_tokens = self.tokenizer(temp, add_special_tokens=False)["input_ids"]
			pointer = tree[2][0]
			for token in temp_tokens:
				if token not in pointer:
					pointer[token] = {}
				pointer = pointer[token]
			pointer[2] = {}
		return tree

	def search(self, subj, obj, subj_type, obj_type, sent):
		tree = self.build_tree(subj, obj, subj_type, obj_type)
		sent = sent.detach().cpu().tolist()
		if sent[-1] == 1:
			return [1]
		else:
			for token in sent:
				if token in tree:
					tree = tree[token]
				else:
					break
			next_token = list(tree.keys())
			return next_token


class PredictTrieNode(object):
	def __init__(
		self,
		prob=None,
		rel=None
	):
		self.prob = prob
		self.rel = rel
		self.next_token = {}

	@property
	def children_num(self):
		return len(self.next_token)

	@property
	def children(self):
		return list(self.next_token.keys())

	def __contains__(self, key):
		return key in self.next_token

	def __call__(self, key):
		return self.next_token[key]

	def __str__(self):
		return f"[{self.prob}|{self.rel}|str({self.next_token})]"

	def __repr__(self):
		return f"[{self.prob}|{self.rel}|str({self.next_token})]"


class PredictTrie(object):
	def __init__(
		self,
		tokenizer,
		force_end=True
	):
		self.tokenizer = tokenizer
		self.force_end = force_end

	def build_tree(self, templates):
		trie = PredictTrieNode()
		for rel, temp in templates.items():
			temp_tokens = self.tokenizer(temp, add_special_tokens=False)["input_ids"]
			if self.force_end:
				temp_tokens = [2, 0] + temp_tokens + [2]
			else:
				temp_tokens = [0] + temp_tokens

			pointer = trie
			for token in temp_tokens:
				if token not in pointer:
					pointer.next_token[token] = PredictTrieNode()
				pointer = pointer(token)
			pointer.rel = rel
		return trie

	def predict(self, subj, obj, raw_templates, inputs, model):
		templates = {}
		for rel in raw_templates:
			templates[rel] = raw_templates[rel].format(subj=subj, obj=obj)
		self.trie = self.build_tree(templates)
		self.trie.prob = 1
		queue = [(self.trie, [])]
		result = {}
		while queue:
			node, path = queue.pop(0)
			if node.children_num > 0:
				for child in node.children:
					queue.append((node(child), path + [child]))
				if node.children_num == 1:
					for child in node.children:
						node(child).prob = node.prob
				else:
					decoder_inputs = torch.LongTensor([path]).to(inputs["input_ids"].device)
					logits = self.get_predict_logit(model, inputs, decoder_inputs)
					score = torch.softmax(logits[node.children], 0)
					if False:
						print(logits[16], logits[34])
						print(path)
						print(node.children)
						print(self.tokenizer.batch_decode(path, skip_special_tokens=True))
						print(score, self.tokenizer.batch_decode(node.children))
					for i, child in enumerate(node.children):
						node(child).prob = node.prob * float(score[i])
			else:
				result[node.rel] = node.prob
		return result

	def get_predict_logit(self,
		model,
		encoder_inputs,
		decoder_inputs,
	):
		with torch.no_grad():
			generate_output = model(
				input_ids=encoder_inputs["input_ids"],
				attention_mask=encoder_inputs["attention_mask"],
				decoder_input_ids=decoder_inputs,
				return_dict=True)
			logits = generate_output.logits[0, -1, :].cpu()
		return logits


class BatchPredictTrie(object):
	def __init__(
		self,
		tokenizer,
		templates
	):
		self.tokenizer = tokenizer
		self.templates = templates
		self.trie = self.build_tree(templates)
		self.relations = list(templates.keys())
		self.relation_path, self.path_set = self.find_path()

	def build_tree(self, templates):
		trie = PredictTrieNode()
		for rel, temp in templates.items():
			pointer = trie
			for token in temp.split(" "):
				if token not in pointer:
					pointer.next_token[token] = PredictTrieNode()
				pointer = pointer(token)
			pointer.rel = rel
		return trie

	def find_path(self):
		queue = [("", self.trie, "")]
		result = {}
		while queue:
			word, node, path = queue.pop(-1)
			path += word
			path += " "
			if node.children_num > 1:
				path += "[SEP] "
			if node.children_num > 0:
				for child in node.children:
					queue.append((child, node(child), path))
			else:
				result[node.rel] = path
		path_set = set()
		for key, value in result.items():
			path = [[[], None]]
			temp = value.strip().split(" ")
			for i in range(len(temp)):
				if temp[i] == "[SEP]":
					path[-1][1] = temp[i+1]
					path_set.add(tuple(path[-1][0]))
					path.append([[], None])
					path[-1][0] += path[-2][0]
				else:
					path[-1][0].append(temp[i])
			result[key] = path[:-1]
		return result, list(path_set)

	def predict(self, subj, obj, sentence, model, device):
		encoder_inputs = self.tokenizer.batch_encode_plus([sentence]*len(self.path_set), return_tensors="pt",
			max_length=256, padding='max_length', truncation=True).to(device)
		decoder_sents = []
		for each in self.path_set:
			decoder_sents.append(" ".join(each).format(subj=subj, obj=obj))
		decoder_inputs = self.tokenizer.batch_encode_plus(decoder_sents, return_tensors="pt", max_length=64, padding='max_length', truncation=True).to(device)
		logits = self.get_predict_logit(model, encoder_inputs, decoder_inputs)
		print(logits.size())

	def get_predict_logit(self,
		model,
		encoder_inputs,
		decoder_inputs,
	):
		with torch.no_grad():
			generate_output = model(
				input_ids=encoder_inputs["input_ids"],
				attention_mask=encoder_inputs["attention_mask"],
				decoder_input_ids=decoder_inputs["input_ids"],
				decoder_attention_mask=decoder_inputs["attention_mask"],
				return_dict=True)
			logits = generate_output.logits.cpu()
			#logits = generate_output.logits[0, -1, :].cpu()
		return logits


if __name__ == "__main__":
	config = "facebook/bart-large-cnn"
	device = torch.device("cuda:3")
	tokenizer = AutoTokenizer.from_pretrained(config)
	model = BartForConditionalGeneration.from_pretrained(config).to(device)
	model.eval()
	subj = "Head entity"
	obj = "Tail entity"

	with open("./data/templates/tacred/rel2temp.json") as f:
		templates = json.load(f)

	#trie = PredictTrie(subj, obj, templates, tokenizer)
	#trie.predict(None, None)

	trie = BatchPredictTrie(tokenizer, templates)
	trie.predict("subj", "obj", "this is sentence", model, device)
