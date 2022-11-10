import json
import os
import torch
import transformers
from utils.trie import Trie, PredictTrie
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score


class Prefix_fn_cls_tree(object):
	def __init__(self, tokenizer, templates, type_constraint=None):
		self.trie = Trie(templates, tokenizer, type_constraint)

	def get(self, batch_id, sent, batch_entities, batch_types):
		subj, obj = batch_entities[batch_id]
		subj_type, obj_type = batch_types[batch_id]
		outputs = self.trie.search(subj, obj, subj_type, obj_type, sent)
		return outputs


def load_inputs(input_path):
	data = []
	with open(input_path) as f:
		line = f.readline()
		while line:
			data.append(json.loads(line))
			line = f.readline()
	return data


if __name__ == "__main__":
	device = torch.device("cuda:3")
	dataset = "tacrev"
	data_version = "v7"
	split = "test"
	model_name = "bart-large-xsum"
	config = f"output/pretrained/pretrained_model_{dataset}_{data_version}_{model_name}"
	#model_name = "pegasus-large-zeroshot"
	#config = "google/pegasus-large"

	input_path = f"./data/{dataset}/{data_version}/{split}.json"
	if data_version == "v10":
		template_path = f"data/templates/{dataset}/rel2temp_na_two_entities.json"
	elif "v11" in data_version:
		template_path = f"data/templates/{dataset}/rel2temp_raw_relation.json"
	elif "v12" in data_version:
		template_path = f"data/templates/{dataset}/rel2temp_forward.json"
	else:
		template_path = f"data/templates/{dataset}/rel2temp.json"
	type_constraint_path = f"data/{dataset}/v1/type_constraint.json"
	type_constraint = True
	output_path = f"output/scoring/{dataset}_{split}_{data_version}_{'trie_type_constraint' if type_constraint else 'no_type_constraint'}_{model_name}.json"

	tokenizer = AutoTokenizer.from_pretrained(config)
	trie = PredictTrie(tokenizer, force_end="bart" in config)
	model = AutoModelForSeq2SeqLM.from_pretrained(config).to(device)
	model.eval()

	with open(template_path) as f:
		templates = json.load(f)

	if type_constraint:
		with open(type_constraint_path) as f:
			type_constraint_dict = json.load(f)


	data =  load_inputs(input_path)
	results = []
	for each in tqdm(data):
		idx = each["id"]
		subj = each["subj"]
		obj = each["obj"]
		true = each["relation"]
		sentence = each["text"]

		inputs = tokenizer.batch_encode_plus([sentence], return_tensors="pt",
			max_length=256, padding='max_length', truncation=True).to(device)

		if type_constraint:
			subj_type = each["subj_type"]
			obj_type = each["obj_type"]
			temp = {"no_relation": templates["no_relation"]}
			for rel in templates:
				if rel in type_constraint_dict[f"{subj_type}|{obj_type}"]:
					temp[rel] = templates[rel]
		else:
			temp = templates

		score = trie.predict(subj, obj, temp, inputs, model)
		p = sorted(score.items(),key=lambda x: -x[1])
		if p[0][0] == "no_relation":
			p_score = p[1]
		else:
			p_score = p[0]
		na_score = score["no_relation"] if "no_relation" in score else score["Other"]
		results.append((p_score, na_score, true))

	with open(output_path, "w") as f:
		json.dump(results, f)
