import json
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description = 'calculate F1 score from scoring files')
	parser.add_argument('--input_file_path', type=str, help="path of the input scoring file")
	parser.add_argument('--template_file_path', type=str, help="path of the template file")
	args = parser.parse_args()

	with open(args.input_file_path) as f:
		data = json.load(f)

	with open(args.template_file_path) as f:
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
