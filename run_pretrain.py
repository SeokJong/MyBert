import json

import tensorflow as tf
from tensorflow.keras import Model

from model import models, data

if __name__ == "__main__":
    with open("base_model.json") as fin:
        options = json.load(fin)
    data_pattern = f"./{options['PRETRAINING_DIR']}/*"
    input_files = []
    input_files.extend(tf.io.gfile.glob(data_pattern))
    builder = data.input_fn_builder(input_files, 128, 20, True, 4)
    data = builder(options)
    model = models.BertPretrainModel(options, tf.keras.activations.gelu)
    model.compile("adam")
    for i in data:
        break
    model.fit(i)


