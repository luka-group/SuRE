import json
from collections import defaultdict

with open("../data/docred/v1/train.json") as f:
	data = [json.loads(line) for line in f.readlines()]

type_dict = set()
type_constraint = defaultdict(set)

for each in data:
	type_dict.add(each["subj_type"])
	type_dict.add(each["obj_type"])
	if each["relation"] != "no_relation":
		type_constraint[f'{each["subj_type"]}|{each["obj_type"]}'].add(each["relation"])

with open("../data/docred/v1/type.json", "w") as f:
	json.dump(list(type_dict), f)

for key in type_constraint:
	type_constraint[key] = list(type_constraint[key])
with open("../data/docred/v1/type_constraint.json", "w") as f:
	json.dump(dict(type_constraint), f)
