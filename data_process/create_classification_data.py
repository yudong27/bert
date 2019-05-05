from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import Counter, defaultdict
import random
import tensorflow as tf
from gensim.models import Word2Vec
import json

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input es text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None, "the word to index save path")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 32, "Maximum sequence length.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_string("w2v_file",None, "word to vector model path")

flags.DEFINE_string("field",None, "which field in corpus to be tokenized, could be question, question_seg, question_method...")

flags.DEFINE_string("tokenization_method",'space',"How to segment sentence, could be space, single_word")



def write_instance_to_example_files(instances, max_seq_length, vocab_index, faq_num, output_files):
    """Create TF example files from `TrainingInstance`s."""
    def create_int_feature(values):
        feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return feature
    def create_float_feature(values):
        feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return feature


    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = [vocab_index[w] if w in vocab_index  else vocab_index['<UNK>'] for w in instance.tokens]

        while len(input_ids) < max_seq_length:
            input_ids.append(vocab_index['<PAD>'])
        input_ids = input_ids[:max_seq_length]
        assert len(input_ids) == max_seq_length

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        #onehot = [0.0 if i!=instance.faq_id else 1.0 for i in range(faq_num)]
        features['faq_id'] = create_int_feature([instance.faq_id])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info(instance.tokens)
            tf.logging.info(instance.faq)
            tf.logging.info(instance.faq_id)
            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info( "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)



def vocab_build(w2v_model_path):
    word2vec_model = Word2Vec.load(w2v_model_path)
    embedding_dim = word2vec_model.vector_size
    tf.logging.info("embedding size = {}".format(embedding_dim))
    vocab_index = dict()
    #embedding_mat = np.zeros( (len(w2v_model.wv.vocab)+2, embedding_dim), dtype=np.float32)
    vocab_index['<PAD>'] = 0
    vocab_index['<UNK>'] = 1
    for idx, word in enumerate(word2vec_model.wv.vocab.keys()):
        word_idx = idx + 2
        #embedding_mat[word_idx, ] = w2v_model[word]
        vocab_index[word] = word_idx
    tf.logging.info("vocab size = {}".format(len(vocab_index)))

    with open(FLAGS.vocab_file,'w' )as f:
        f.write(json.dumps(vocab_index,ensure_ascii=False, indent=3))
    return vocab_index

class TrainingInstance(object):
    """A single training instance (sentence pair)."""
    def __init__(self, tokens, faq, faq_id):
        self.tokens = tokens
        self.faq = faq
        self.faq_id = faq_id

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(self.tokens))
        s += "faq: %s\n"% (self.faq)
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

def create_instances_from_document(corpus, max_seq_length, vocab_words, faq2id, field, tokenization_method):
    """Creates `TrainingInstance`s for a single document."""
    assert field in corpus
    assert tokenization_method in ['space','single_word']
    instances = []
    if tokenization_method == 'space':
        tokens = corpus[field].split(' ')
    elif tokenization_method == 'single_word':
        tokens = [w for w in corpus[field]]
    else:
        pass

    instance = TrainingInstance( tokens=tokens, faq=corpus['faq'], faq_id=faq2id[corpus['faq']])
    instances.append(instance)
    return instances



def create_training_instances(input_files, max_seq_length, vocab_words):
    all_corpus = []

    for input_file in input_files:
        with open(input_file, "r") as reader:
            corpus = json.load(reader)
            all_corpus.extend(corpus)
    faq_count = Counter([corpus['faq'] for corpus in all_corpus])
    tf.logging.info("faq num = {}".format(len(faq_count)))
    faq2id = {k:i for i,(k,v) in enumerate(faq_count.most_common())}
    instances = []
    for corpus in all_corpus:
        instances.extend(
          create_instances_from_document(corpus, max_seq_length, vocab_words, faq2id, FLAGS.field, FLAGS.tokenization_method))

    return instances, len(faq_count)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    vocab_words = vocab_build(FLAGS.w2v_file)

    instances, faq_num = create_training_instances( input_files, FLAGS.max_seq_length, vocab_words)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances,  FLAGS.max_seq_length, vocab_words, faq_num, output_files)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("w2v_file")
    flags.mark_flag_as_required("field")
    tf.app.run()
