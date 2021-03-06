export CUDA_VISIBLE_DEVICES=2
python run_pretraining.py \
  --input_file=tf_examples.tfrecord \
  --output_dir=./pretraining_output \
  --do_train=True \
  --do_eval=False \
  --bert_config_file=bert_config.json \
  --train_batch_size=96 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=2000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=False
