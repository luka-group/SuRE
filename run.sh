dataset_path=data
dataset_name=tacred
data_version=v0_0.01
cuda_device_id=3
train_file_name=${data_version}/train.json
valid_file_name=${data_version}/dev.json

model_source=google
pretrain_model=pegasus-large
suffix=_1e5_wd_5e6

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
