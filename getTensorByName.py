import sys
from bert_and_ernie import modeling
import os
import tensorflow as tf
import json
import numpy as np
from tensorflow.contrib import learn


os.environ['CUDA_VISIBLE_DEVICES'] = ''
print(tf.__version__)
model_dir = '../chinese_L-12_H-768_A-12'
config_name = os.path.join(model_dir, "bert_config.json")
vocab_path = os.path.join(model_dir, "vocab.txt")
ckpt_name = "bert_model.ckpt"
max_seq_length = 64

with open(vocab_path, encoding='utf-8') as f:
    data = f.readlines()
    vocab_dict = {x.strip():i for i,x in enumerate(data)}
    idx2vocab = {i:w for w,i in vocab_dict.items()}
    #pprint(idx2vocab)
def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                            bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1,name="log_probs")

#        label_ids = tf.reshape(label_ids, [-1])
#        label_weights = tf.reshape(label_weights, [-1])
#
#        one_hot_labels = tf.one_hot(
#                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
#
#        # The `positions` tensor might be zero-padded (if the sequence is too
#        # short to have the maximum number of predictions). The `label_weights`
#        # tensor has a value of 1.0 for every real prediction and 0.0 for the
#        # padding predictions.
#        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
#        numerator = tf.reduce_sum(label_weights * per_example_loss)
#        denominator = tf.reduce_sum(label_weights) + 1e-5
#        loss = numerator / denominator
    return log_probs


def sentence2tokens(query):
    tokens = []
    tokens.append('[CLS]')
    for w in query:
        tokens.append(w)
    tokens.append('[SEP]')
    return tokens

def tokens2ids(tokens, masked_lm_positions):
    #vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    #vocab_dict = vocab_processor.vocabulary_._mapping
    # addhoc changes to [CLS] [SEP] [MASK]
#   vocab_dict['[CLS]'] = vocab_dict.pop('CLS')
#    vocab_dict['[SEP]'] = vocab_dict.pop('SEP')
#    vocab_dict['[MASK]'] = vocab_dict.pop('MASK')
    
    input_ids = []
    for idx, w in enumerate(tokens):
        if idx in masked_lm_positions[0]:
            input_ids.append(vocab_dict['[MASK]'])
            print(idx, tokens[idx])
        elif w not in vocab_dict:
            input_ids.append(vocab_dict['[UNK]'])
        else:
            input_ids.append(vocab_dict[w])
    input_mask = [1] * len(input_ids)
    segment_ids = [0]* len(input_ids) # 如果有第二句，则第二句为1
    
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)


    ## 一下三个mask用来计算loss，与每个位置的结果无关
    #masked_lm_positions = list(instance.masked_lm_positions)
    #masked_lm_ids = []
    #for w in instance.masked_lm_labels:
    #  if w not in vocab_dict:
    #    masked_lm_ids.append(vocab_dict['<UNK>'])
    #  else:
    #    masked_lm_ids.append(vocab_dict[w])
    #
    #masked_lm_weights = [1.0] * len(masked_lm_ids)
    
    
    #while len(masked_lm_positions) < max_predictions_per_seq:
    #  masked_lm_positions.append(0)
    #  masked_lm_ids.append(0)
    #  masked_lm_weights.append(0.0)
    return  [input_ids], [input_mask], [segment_ids]

init_checkpoint = os.path.join(model_dir, ckpt_name)

def model_builder():
    with tf.gfile.GFile(config_name, 'r') as f:
        bert_config = modeling.BertConfig.from_dict(json.load(f))
    
    input_ids = tf.placeholder(tf.int32, (None, None), 'input_ids')
    input_mask = tf.placeholder(tf.int32, (None, None), 'input_mask')
    input_type_ids = tf.placeholder(tf.int32, (None, None), 'input_type_ids')
    masked_lm_positions = tf.placeholder(tf.int32, (None, None), "masked_lm_positions")
    #masked_lm_ids = tf.placeholder(tf.int32, (None, None), "masked_lm_ids")
    #masked_lm_weights = tf.placeholder(tf.float32, (None, None), "masked_lm_weights")
    
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=False)
    
    
    masked_lm_log_probs = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(), masked_lm_positions)
    
    
    tvars = tf.trainable_variables()
    
    #for tv in tvars:
    #    if "cls/predictions" in tv.name:
    #        print(tv)
    (assignment_map, initialized_variable_names
     ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    
    
    
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    return input_ids, input_mask, input_type_ids, masked_lm_positions, masked_lm_log_probs, model.get_sequence_output()

#g = tf.get_default_graph()

#for op in g.get_operations():
#    if "cls/predictions/log_probs" in op.name:
#        print(op.name, op)

#log_probs_output = g.get_tensor_by_name("cls/predictions/log_probs:0")
def run():
    query = "绝大多数动物都是雌性生育并抚养后代的。"
    tokens = sentence2tokens(query)
    masked_lm_positions_in = [[9,10]]
    print(query)
    print("masked_lm_positions_in",masked_lm_positions_in)
    input_ids_in, input_mask_in, input_type_ids_in = tokens2ids(tokens, masked_lm_positions_in)
    print("input_ids_in",input_ids_in)
    print("input_mask_in",input_mask_in)
    print("input_type_ids_in",input_type_ids_in)
    input_ids, input_mask, input_type_ids, masked_lm_positions, masked_lm_log_probs, sequence_output = model_builder()
    with tf.Session() as sess:
        gv = tf.global_variables()
    #    print(len(gv))
        sess.run(tf.global_variables_initializer())
        output,sq_output = sess.run([masked_lm_log_probs, sequence_output],
            feed_dict={input_ids: input_ids_in,
                       input_mask: input_mask_in,
                       input_type_ids: input_type_ids_in,
                       masked_lm_positions:masked_lm_positions_in,
                       })
        print(output.shape)
        #print(sq_output)
        indx = np.argsort(output, axis=1)
        for x in indx:
            x = x[::-1]
            #print(x[0])
            #print(idx2vocab[x[0]])
            print([idx2vocab[x[i]] for i in range(5)])

run()
