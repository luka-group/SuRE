import json
import os
import torch
import transformers
import argparse
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

	parser = argparse.ArgumentParser(
		description = 'run trie scoring with pretrained summarization model')
	parser.add_argument('--dataset', type=str, help="name of dataset")
	parser.add_argument('--data_version', type=str, help="name of particular data version")
	parser.add_argument('--split', type=str, help="name of data split")
	parser.add_argument('--model_name', type=str, help="name of the backbone summarization model")
	parser.add_argument('--config', type=str, help="config of the checkpoint")
	parser.add_argument('--cuda', type=int, help="cuda index")
	parser.add_argument('--type_constraint', action='store_true')
	args = parser.parse_args()

	for arg in vars(args):
		print(f"{arg}: {getattr(args, arg)}")

	device = torch.device(f"cuda:{args.cuda}")
	input_path = f"./data/{args.dataset}/{args.data_version}/{args.split}.json"
	template_path = f"data/templates/{args.dataset}/rel2temp.json"
	type_constraint_path = f"data/{args.dataset}/types/type_constraint.json"
	output_path = f"output/scoring/{args.dataset}_{args.split}_{args.data_version}_{'trie_type_constraint' if args.type_constraint else 'no_type_constraint'}_{args.model_name}.json"

	tokenizer = AutoTokenizer.from_pretrained(args.config)
	trie = PredictTrie(tokenizer, force_end="bart" in args.config)
	model = AutoModelForSeq2SeqLM.from_pretrained(args.config).to(device)
	model.eval()

	with open(template_path) as f:
		templates = json.load(f)

	if args.type_constraint:
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

		if args.type_constraint:
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
