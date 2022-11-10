import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report


if __name__ == "__main__":
	input_file_path = "output/scoring/tacred_test_v0_trie_type_constraint_pegasus-large_eval_1e4_wd_5e6.json"
	template_file_path = "data/templates/tacred/rel2temp.json"

	with open(input_file_path) as f:
		data = json.load(f)

	with open(template_file_path) as f:
		templates = json.load(f)

	relation_map = {rel:i for i, rel in enumerate(list(templates.keys()))}
	relation_num = len(relation_map)

	trues = []
	preds = []
	for pred, na_score, true in tqdm(data):
		trues.append(relation_map[true])
		if na_score > pred[1]:
			preds.append(relation_map["no_relation"])
		else:
			preds.append(relation_map[pred[0]])

	f1 = f1_score(trues, preds, labels=range(1, relation_num), average="micro")
	print("F1 score:", f1)
	print("Classification report:", classification_report(trues, preds, labels=range(1, relation_num), digits=4))
