import re
import os
import json
from tqdm import tqdm
import argparse


def transform_tacred( dataset,
	input_file_path,
	output_file_path,
	template_file_path,
	zero_shot_idx_file_path=None,
	aug_note=False,
	aug_type=False,
	aug_tag=False,
	type_mask=False,
	aug_type_tag=False,
	aug_ch_tag=False,
	replace_basket_tags=False,
	neg_copy=False
	):

	# loading input file
	print(f"Transforming {input_file_path} to {output_file_path}")
	with open(input_file_path) as f:
		data = json.load(f)

	if zero_shot_idx_file_path:
		with open(zero_shot_idx_file_path) as f:
			zero_shot_idx = set([each.replace("\n", "") for each in f.readlines()])
	else:
		zero_shot_idx = None

	# loading template file
	with open(template_file_path) as f:
		rel2temp = json.load(f)

	outputs = []
	sample_idx = 0
	for each in tqdm(data, ncols=80):
		idx = each["id"]
		relation = each['relation']
		token = each['token']
		subj_start = each["subj_start"]
		subj_end = each["subj_end"]
		subj_type = each["subj_type"]
		obj_start = each["obj_start"]
		obj_end = each["obj_end"]
		obj_type = each["obj_type"]

		if zero_shot_idx and idx not in zero_shot_idx:
			continue

		subj = " ".join(token[subj_start:subj_end+1])
		obj = " ".join(token[obj_start:obj_end+1])

		template = rel2temp[relation]

		if type_mask:
			for i in range(subj_start, subj_end + 1):
				token[i] = subj_type
			for i in range(obj_start, obj_end + 1):
				token[i] = obj_type

		if aug_tag:
			if aug_type_tag:
				token_tagged = []
				for i, each in enumerate(token):
					if i == subj_start:
						token_tagged.append(f"<e1-{subj_type}>")
					if i == subj_end + 1:
						token_tagged.append(f"</e1-{subj_type}>")
					if i == obj_start:
						token_tagged.append(f"<e2-{obj_type}>")
					if i == obj_end + 1:
						token_tagged.append(f"</e2-{obj_type}>")
					token_tagged.append(each)
				sentence = " ".join(token_tagged)
			elif aug_ch_tag:
				token_tagged = []
				for i, each in enumerate(token):
					if i == subj_start:
						token_tagged.append(f"@ * {subj_type.lower()} * ")
					if i == subj_end + 1:
						token_tagged.append("@")
					if i == obj_start:
						token_tagged.append(f"# ^ {obj_type.lower()} ^ ")
					if i == obj_end + 1:
						token_tagged.append("#")
					token_tagged.append(each)
				sentence = " ".join(token_tagged)
			else:
				token_tagged = []
				for i, each in enumerate(token):
					if i == subj_start:
						token_tagged.append("<e1>")
					if i == subj_end + 1:
						token_tagged.append("</e1>")
					if i == obj_start:
						token_tagged.append("<e2>")
					if i == obj_end + 1:
						token_tagged.append("</e2>")
					token_tagged.append(each)
				sentence = " ".join(token_tagged)
		else:
			sentence = " ".join(token)

		if type_mask:
			sentence = re.sub(f"<e1>(.*)</e1>", f"<e1> {subj_type} </e1>", sentence)
			sentence = re.sub(f"<e2>(.*)</e2>", f"<e2> {obj_type} </e2>", sentence)

		if aug_type:
			head_type_sentence = f"The type of {subj} is {subj_type.lower()} . "
			tail_type_sentence = f"The type of {obj} is {obj_type.lower()} . "
			sentence = head_type_sentence + tail_type_sentence + sentence

		if aug_note:
			head_sentence = f"The head entity is {subj} . "
			tail_sentence = f"The tail entity is {obj} . "
			sentence = head_sentence + tail_sentence + sentence

		if type_mask:
			target = template.format(subj=subj_type, obj=obj_type)
		else:
			target = template.format(subj=subj, obj=obj)

		if replace_basket_tags:
			subj = subj.replace("-LRB-", "(").replace("-RRB-", ")")
			subj = subj.replace("-LSB-", "[").replace("-RSB-", "]")
			obj = obj.replace("-LRB-", "(").replace("-RRB-", ")")
			obj = obj.replace("-LSB-", "[").replace("-RSB-", "]")
			target = target.replace("-LRB-", "(").replace("-RRB-", ")")
			target = target.replace("-LSB-", "[").replace("-RSB-", "]")
			sentence = sentence.replace("-LRB-", "(").replace("-RRB-", ")")
			sentence = sentence.replace("-LSB-", "[").replace("-RSB-", "]")

		if neg_copy and relation == "no_relation":
			target = sentence

		record = {
			"id": f"{input_file_path}_{dataset}_{sample_idx}",
			"text": sentence,
			"target": target,
			"subj": subj,
			"subj_type": subj_type,
			"obj": obj,
			"obj_type": obj_type,
			"relation": relation}
		outputs.append(record)
		sample_idx += 1

	with open(output_file_path, "w") as fo:
		for record in outputs:
			fo.write(json.dumps(record)+"\n")

if __name__ == "__main__":
	dataset = ("tacred",)
	data_version = "v0"
	low_resource = True # process the low resource data
	proc_kwargs = {
		"v0": {
			"aug_note": True,
			"aug_type": True,
		}
	}

	if low_resource:
		zero_shot_file_dir = "../data/tacred/tacred_splits/"
		if "tacred" in dataset:
			template_file_path = "../data/templates/tacred/rel2temp.json"
			for split in ("train", "dev"):
				for prop in ("0.01", "0.05", "0.1"):
					zero_shot_idx_file_path = os.path.join(zero_shot_file_dir, split,
						f"{prop}.split.txt" if split == "train" else f"dev.{prop}.split.txt")
					transform_tacred(
						dataset = "tacred",
						input_file_path = f"../data/tacred/data/json/{split}.json",
						output_file_path = f"../data/tacred/{data_version}_{prop}/{split}.json",
						template_file_path = template_file_path,
						zero_shot_idx_file_path = zero_shot_idx_file_path,
						**proc_kwargs[data_version]
					)
	else:
		if "tacred" in dataset:
			template_file_path = "../data/templates/tacred/rel2temp.json"
			for split in ("train", "dev", "test"):
				transform_tacred(
					dataset = "tacred",
					input_file_path = f"../data/tacred/data/json/{split}.json",
					output_file_path = f"../data/tacred/{data_version}/{split}.json",
					template_file_path = template_file_path,
					**proc_kwargs[data_version]
				)
