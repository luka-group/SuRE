from dataclasses import dataclass
from typing import List
import json


@dataclass
class SummaryFeature:
	'''sample feature for finetuning summarization model'''	
	_id: str
	text: str
	target: str
	subj: str
	obj: str
	relation: str
	subj_type: str = None
	obj_type: str = None

	def get_sample_id(self) -> int:
		return int(self._id.split("_")[1])

	def get_template_id(self) -> int:
		return int(self._id.split("_")[2])

	def get_type_constraint(self) -> str:
		return f"{self.subj_type}:{self.obj_type}"


def load_summary_features(path: str) -> List[SummaryFeature]:
	features = []
	with open(path) as f:
		line = f.readline()
		while line:
			data = json.loads(line)
			feature = SummaryFeature(
				_id = data["id"],
				text = data["text"],
				target = data["target"],
				subj = data["subj"],
				obj = data["obj"],
				relation = data["relation"],
				subj_type = data["subj_type"] if "subj_type" in data else None,
				obj_type = data["obj_type"] if "obj_type" in data else None
			)
			features.append(feature)
			line = f.readline()
	return features
