python create_classification_data.py \
  --input_file ../../data/training/train_data_20190412_es.json \
  --output_file 0412.tfrecord \
  --w2v_file ../../data/model_required_data/faq_w2v200_jieba_word_0312_fullmode.model \
  --vocab_file word2index.txt \
  --field question_seg
