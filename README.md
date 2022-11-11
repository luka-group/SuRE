# Summarization as Indirect Supervision for Relation Extraction

Authors: Keming Lu, I-Hung Hsu, Wenxuan Zhou, Mingyu Derek Ma, Muhao Chen

:tada: This work is accepted by Findings of EMNLP2022 :tada:

- [Nov. 10th 2022] We have shared codes for training and inference of SuRE. Come and try SuRE on TACRED in both full-training and low-resource settings or your customized datasets!

## Overview

This repository provides codes of SuRE (Summarization as Relation Extraction). SuRE converts RE into a summarization formulation. SURE leads to more precise and resource-efficient RE based on indirect supervision from summarization tasks. To achieve this goal, we develop sentence and relation conversion techniques that essentially bridge the formulation of summarization and RE tasks. We also incorporate constraint decoding techniques with Trie scoring to further enhance summarization-based RE with robust inference. Experiments on three RE datasets demonstrate the effectiveness of SURE in both full-dataset and low-resource settings, showing that summarization is a promising source of indirect supervision to improve RE models.

## Key Idea of SuRE

The summarization task takes a context as the input sequence and a summary target is expected to be generated. To formulate RE as summarization, we first need to hint the summarization model which entity pair is targeted for summarization. To do so, we process the input sentence such that entity mentions and their type information will be highlighted. We explore existing entity marking tricks and also develop entity information verbalization technique that directly augments entity information as part of the context. The processed sentence will then be fed into SuRE. The summary targets for SuRE is created via verbalizing existing RE labels to templates, such as the Relation Verbalization subfigure in Fig. 1. In the training process, SuRE uses pretrained summarization models as a start point, and finetunes them with processed sentences as the input and verbalized relation descriptions as the targets. During inference, we incorporate several constrained inference techniques to help SURE decide the inferred relation

![Figure 1. Example for SuRE Inference](https://github.com/luka-group/SuRE/blob/main/figure1.png)


## Requirements

SuRE is tested to work under Python 3.8+. We trained and evaluated SuRE on 4 NVIDIA RTX A5000 with CUDA VERSION 11.3. Packages required include

- accelerate==0.5.1
- datasets==1.12.1
- filelock==3.3.0
- huggingface_hub==0.0.19
- matplotlib==3.5.0
- nltk==3.6.3
- numpy==1.20.3
- rouge==1.0.1
- scikit_learn==1.1.0
- torch==1.10.2+cu113
- tqdm==4.62.3
- transformers==4.11.3

All packages can be installed by `pip install -r requirements.txt`

## Data

### Relation Templates

We collect relation templates for TACRED and TACREV from [Sainz et al.](https://github.com/osainz59/Ask2Transformers) and do manual refinements. We also manually constructed relation templates for other popular RE datasets, such as DocRED, TACREV, ReTACRED, and Semeval.

```
data/templates
- tacred: relation templates for TACRED
  - rel2temp.json: relation templates used in the main result of our paper
  - rel2temp_forward.json: another semantic templates for ablation study
  - rel2temp_na_two_entities.json: relation templates for ablation study of the NA template
  - rel2temp_raw_relation.json: naive structural templates for ablation study
- docred: relation templates for DocRED
- retacred: relation templates for ReTACRED
- tacrev: relation templates for TACREV
- semeval: relation templates for SemEVAL
```

Formats of template files: JSON files with relation names as keys and templates as values. {subj} and {obj} are placeholders for head and tail entities. Customized relation template files in the same format can easily be used in our codes.

```
{
  "no_relation": "{subj} has no known relations to {obj}", # We use this template for the NA relation in all datasets
}
```

## Processed TACRED data

```
data/tacred/
- tacred_splits: indice of tacred samples for the low-resource training (1%/5%/10%) from [Sainz et al.](https://github.com/osainz59/Ask2Transformers)
- types: type-related auxiliary data in TACRED
  - type.json: a list of entity type in JSON format
  - type_constraint.json: a mapping between head|entity types and feasible relations
- v0: processed full-training TACRED dataset
- v0.01: processed TACRED dataset under the 1% scenario
- v0.05: processed TACRED dataset under the 5% scenario
- v0.1: processed TACRED dataset under the 10% scenario
```

Data format:
- Type constraint files:
```
{
  # "type_name1|type_name2": [relation_name1, ...] type_names should be the same in the type.json file
  "ORGANIZATION|PERSON": [
        "org:top_members/employees",
        "org:shareholders",
        "org:founded_by"
    ],
}
```
- Data samples:
```
{
  "id": "some index", # index for each sample
  "text": "The head entity is Douglas Flint . The tail entity is chairman . The type of Douglas Flint is person . The type of chairman is title . At the same time , Chief Financial Officer Douglas Flint will become chairman , succeeding Stephen Green who is leaving to take a government job .",
  # input text: augmented head entity mention + augmented tail entity mention + augmented head entity type + augmented tail entity type + context
  "target": "Douglas Flint is a chairman", # target summary generated by relation templates '{subj} is a {obj}'
  "subj": "Douglas Flint", # subject mention
  "subj_type": "PERSON", # subject entity type
  "obj": "chairman", # object mention
  "obj_type": "TITLE", # object entity type
  "relation": "per:title" # Relation name
}
```

## Run

### Inference

Download zip files of pretrained checkpoints from Google Drive:
- TACRED full training: [Google Drive Link](https://drive.google.com/file/d/1e6naPQrL063AqtD3Q-kdrIKDPybhkZEt/view?usp=sharing)
- TACRED low resource (1\%): WIP
- TACRED low resource (5\%): WIP
- TACRED low resource (10\%): WIP

Run inference with Trie constrained decoding with script `predict_trie.sh` by setting following parameters in the script:

```
python -u predict_trie.py \
	--dataset tacred \ # dataset name
	--data_version v0 \ # dataset version (v0: full/v0_0.01: 1%/v0_0.05: 5%/v0_0.1: 10%)
	--split test \ # dataset split: (train/dev/test)
	--model_name pegasus-large\ # model name
	--cuda 3 \ # cuda index
	--type_constraint \ # use type constraint or not
	--config output/pretrained/pretrained_model_tacred_v0_pegasus-large_eval_1e4_wd_5e6 # path to model checkpoint
```

Run scoring to calculate F1 scores from prediction outputs with script `score.sh` by setting following parameters in the script:
```
python score.py \
	--input_file_path output/scoring/tacred_test_v0_trie_type_constraint_pegasus-large.json \ # predict output from predict_trie.sh
	--template_file_path data/templates/tacred/rel2temp.json # relation template file of the corresponding dataset
```

### Training

Run training on specific datasets with script `run.sh` by setting following parameters in the script:

```
dataset_path=data # directory of data
dataset_name=tacred # dataset name
data_version=v0_0.01 # specific data version: (v0: full/v0_0.01: 1%/v0_0.05: 5%/v0_0.1: 10%)
cuda_device_id=3 # cuda index
train_file_name=${data_version}/train.json # training file path
valid_file_name=${data_version}/dev.json # validation file path

# huggingface config: model_source/pretrain_model
model_source=google
pretrain_model=pegasus-large
suffix=_1e5_wd_5e6 # customized suffix in output checkpoint names

checkpoint_name=pretrained_model_${dataset_name}_${data_version}_${pretrain_model}_eval${suffix}

CUDA_VISIBLE_DEVICES=$cuda_device_id python -u run_pretrained_aug_tag_eval.py\
  --model_name_or_path $model_source/$pretrain_model\
	--train_file $dataset_path/$dataset_name/$train_file_name\
	--validation_file $dataset_path/$dataset_name/$valid_file_name\
	--type_file $dataset_path/$dataset_name/types/type.json\
	--type_constraint_file $dataset_path/$dataset_name/types/type_constraint.json\
	--template_file $dataset_path/templates/$dataset_name/rel2temp.json \
	--text_column text \
	--summary_column target \
	--max_source_length 256 \
	--min_target_length 0 \
	--max_target_length 64 \
	--learning_rate 1e-5 \
	--weight_decay 5e-6 \
	--num_beams 4 \
	--num_train_epochs 100 \
	--preprocessing_num_workers 8 \
	--output_dir ./output/pretrained/$checkpoint_name\
	--per_device_train_batch_size=12 \
	--per_device_eval_batch_size=12 \
	--gradient_accumulation_steps 2 \
	--num_warmup_steps 0 \
	--seed 100
```

### How to run SuRE on my customized datasets :question:

We provide an [example data preprocessing script](https://github.com/luka-group/SuRE/blob/main/data_scripts/transform_tacred.py) that we used to transform raw data of TACRED into summarization data.

In general, you need to preprocess your relation extraction datasets with following steps to run training and inference of SuRE on customized datasets.
- Augment context with entity mentions following the template `The head entity is {subj} . The tail entity is {obj} . `
- Augment context with entity types following the template `The type of {subj} is {subj_type.lower()} . The type of {obj} is {obj_type.lower()} . `[optional/only if entity types are available]
- Build a mapping between relations and semantic relation templates
- Transform relations in samples to summarization targets
- Make sure data looks the same as the example in the same format described in the Data section

Following parts are optional, you can still run SuRE on your datasets without these files after simply modifying arguments or codes
- No entity types: Not augment context with entity types in preprocessing
- No type constraints: Not apply type constraints in `predict_trie.sh`

## Citing

```
@article{lu2022summarization,
  title={Summarization as Indirect Supervision for Relation Extraction},
  author={Lu, Keming and Hsu, I and Zhou, Wenxuan and Ma, Mingyu Derek and Chen, Muhao and others},
  journal={arXiv preprint arXiv:2205.09837},
  year={2022}
}
```

## License

SuRE is licensed under the MIT License.
