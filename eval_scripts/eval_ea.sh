python metrics/create_ea_data.py \
  --gold_file data/dev.tsv \
  --eval_annotated_file data/annotations.tsv \
  --output_EA_initial data/ea_initial.tsv \
  --output_EA_final data/ea_final.tsv

python metrics/run_ea.py \
  --model_name_or_path ./models/ea-metric \
  --task_name stance \
  --do_eval \
  --save_steps 1000000 \
  --data_dir ./data/ea_initial.tsv \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 6.0 \
  --output_dir ./models/ea-metric \
  --cache_dir ./models/ \
  --logging_steps 500 \
  --evaluation_strategy="epoch" \
  --overwrite_cache

mv ./models/ea-metric/probs.txt ./models/ea-metric/probs_initial.txt

python metrics/run_ea.py \
  --model_name_or_path ./models/ea-metric \
  --task_name stance \
  --do_eval \
  --save_steps 1000000 \
  --data_dir ./data/ea_final.tsv \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 6.0 \
  --output_dir ./models/ea-metric \
  --cache_dir ./models/ \
  --logging_steps 500 \
  --evaluation_strategy="epoch" \
  --overwrite_cache

mv ./models/ea-metric/probs.txt ./models/ea-metric/probs_final.txt

python metrics/compute_ea.py \
  --initial_probs ./models/ea-metric/probs_initial.txt \
  --final_probs ./models/ea-metric/probs_final.txt \
  --initial_file ./data/ea_initial.tsv \
  --final_file ./data/ea_final.tsv \
  --gold_file data/dev.tsv \

