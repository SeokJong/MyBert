import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense

import bert
import utils


class MaskedLanguageModel(Layer):
    def __init__(self, model_opt: dict, **kwargs):
        super(MaskedLanguageModel, self).__init__(**kwargs)
        self.hidden_size = model_opt["HIDDEN_SIZE"]
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=model_opt["INIT_RANGE"])
        self.hidden_activation = model_opt["HIDDEN_ACTIVATION"]
        self.vocab = model_opt["VOCAB_SIZE"]
        self.prediction = Dense(
            units=self.hidden_size,
            kernel_initializer=self.initializer,
            name="mlm_dense",
            activation=self.hidden_activation
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=model_opt["NORM_EPS"], name="LayerNorm")
        self.ouput_bias = self.add_weight(
            shape=[self.vocab],
            initializer=tf.zeros_initializer()
        )

    def call(self, input_tensor, output_weight, positions, label_ids, label_weights, **kwargs):
        input_tensor = utils.gather_indexes(input_tensor, positions)
        input_tensor = self.prediction(input_tensor)
        logits = tf.matmul(input_tensor, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.ouput_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=self.vocab, dtype=tf.float32
        )

        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

        return (loss, per_example_loss, log_probs)


class NextSentencePrediction(Layer):
    def __init__(self, model_opt: dict, **kwargs):
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=model_opt["INIT_RANGE"])
        self.hidden_size = model_opt["HIDDEN_SIZE"]
        self.pooled = Dense(
            self.hidden_size,
            activation=tf.tanh,
            kernel_initializer=self.initializer,
            name="pooling_layer"
        )
        self.prediction = Dense(
            units=2,
            kernel_initializer=self.initializer,
            name="nsp_dense"
        )
        super(NextSentencePrediction, self).__init__(**kwargs)

    def call(self, inputs, labels, **kwargs):
        first_token_tensor = tf.squeeze(inputs[:, 0:1, :], axis=1)
        pooled_output = self.pooled(first_token_tensor)
        logits = self.prediction(pooled_output)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


class BertPretrainModel(Model):
    def __init__(self, model_opt: dict, mid_activation, **kwargs):
        super(BertPretrainModel, self).__init__(**kwargs)
        self.hidden_size = model_opt["HIDDEN_SIZE"]
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=model_opt["INIT_RANGE"])
        self.bert = bert.Bert(model_opt, mid_activation, name="bert_main", **kwargs)
        self.mlm = MaskedLanguageModel(model_opt, name="mlm", **kwargs)
        self.nsp = NextSentencePrediction(model_opt, name="nsp", **kwargs)


    def call(self, inputs, training=None, **kwargs):
        input_shape = inputs["input_mask"].shape
        extended_attention_mask = tf.reshape(inputs["input_mask"], (input_shape[0], 1, 1, input_shape[1]))
        extended_attention_mask = tf.cast(tf.equal(extended_attention_mask, 0), tf.float32)
        extended_attention_mask = tf.multiply(extended_attention_mask, tf.constant(-10000.0, dtype=tf.float32))
        output = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=extended_attention_mask,
            token_type_ids=inputs["segment_ids"],
            training=training
        )
        loss_mlm = self.mlm(
            input_tensor=output,
            output_weight=self.bert.get_embedding_weight(),
            positions=inputs["masked_lm_positions"],
            label_ids=inputs["masked_lm_ids"],
            label_weights=inputs["masked_lm_weights"]
        )
        loss_nsp = self.nsp(
            inputs=output,
            labels=inputs["next_sentence_labels"]
        )
        total_loss = loss_mlm[0] + loss_nsp[0]
        self.add_loss(total_loss)
        return total_loss

    def save_pretrained(self, save_directory, saved_model=False, version=1):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        :func:`~transformers.TFPreTrainedModel.from_pretrained` class method.
        Arguments:
            save_directory (:obj:`str`):
                Directory to which to save. Will be created if it doesn't exist.
            saved_model (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If the model has to be saved in saved model format as well or not.
            version (:obj:`int`, `optional`, defaults to 1):
                The version of the saved model. A saved model needs to be versioned in order to be properly loaded by
                TensorFlow Serving as detailed in the official documentation
                https://www.tensorflow.org/tfx/serving/serving_basic
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)

        if saved_model:
            saved_model_dir = os.path.join(save_directory, "saved_model", str(version))
            self.save(saved_model_dir, include_optimizer=False, signatures=self.serving)
            logger.info(f"Saved model created in {saved_model_dir}")

        # Save configuration file
        self.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, TF2_WEIGHTS_NAME)
        self.save_weights(output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))
