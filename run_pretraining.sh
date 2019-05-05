export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4\
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
python run_pretraining.py \
  --input_file=tf_examples.tfrecord \
  --output_dir=./pretraining_output \
  --do_train=True \
  --do_eval=False \
  --bert_config_file=bert_config.json \
  --train_batch_size=256 \ #单gpu的batch_size, 不是总的
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=2000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=False
