import tensorflow as tf
filename = "tf_examples.tfrecord"
filename = "0412.tfrecord"

for serialized_example in tf.python_io.tf_record_iterator(filename):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    print(type(example.features.feature))
    for k in example.features.feature.keys():
        print(type(example.features.feature[k]))
    #print(example.features.feature.keys())
    break
    #image = example.features.feature['image'].bytes_list.value
    #label = example.features.feature['label'].int64_list.value
    # 可以做一些预处理之类的
    #print image, label
