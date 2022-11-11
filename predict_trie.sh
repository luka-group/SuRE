python -u predict_trie.py \
	--dataset tacred \
	--data_version v0 \
	--split test \
	--model_name pegasus-large \
	--cuda 3 \
	--type_constraint \
	--config output/pretrained/pretrained_model_tacred_v0_pegasus-large_eval_1e4_wd_5e6
