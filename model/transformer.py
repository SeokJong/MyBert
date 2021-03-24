import math

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class SelfAttention(Layer):
    def __init__(self, model_opt: dict, **kwargs):
        if model_opt["HIDDEN_SIZE"] % model_opt["ATT_HEAD"] != 0:
            raise ValueError(
                f'The hidden size ({model_opt["HIDDEN_SIZE"]}) is not a multiple of the number '
                f'of attention heads ({model_opt["ATT_HEAD"]})'
            )

        self.num_attention_heads = model_opt["ATT_HEAD"]
        self.attention_head_size = int(model_opt["HIDDEN_SIZE"] / model_opt["ATT_HEAD"])
        self.all_head_size = model_opt["HIDDEN_SIZE"]
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=model_opt["INIT_RANGE"])
        self.dropout = tf.keras.layers.Dropout(rate=model_opt["DROPOUT_RATE"], name="DO")

        self.query = tf.keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=self.initializer,
            name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=self.initializer,
            name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=self.initializer,
            name="value"
        )
        super(SelfAttention, self).__init__(**kwargs)

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
            self,
            hidden_states: tf.Tensor,
            attention_mask: tf.Tensor,
            training: bool = False,
    ):
        batch_size = hidden_states.shape[0]
        mixed_query_layer = self.query(inputs=hidden_states)
        mixed_key_layer = self.key(inputs=hidden_states)
        mixed_value_layer = self.value(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        attention_scores = tf.add(attention_scores, attention_mask)

        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)

        attention_probs = self.dropout(inputs=attention_probs, training=training)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        return attention_output


class ResConnLayerNorm(Layer):
    def __init__(self, model_opt: dict, **kwargs):
        self.hidden_size = model_opt["HIDDEN_SIZE"]
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=model_opt["INIT_RANGE"])
        self.att_output = Dense(
            units=self.hidden_size,
            kernel_initializer=self.initializer,
            name="RCLN_in"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=model_opt["NORM_EPS"], name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=model_opt["DROPOUT_RATE"], name="DO")
        super(ResConnLayerNorm, self).__init__(**kwargs)

    def call(self, hidden_states, input_tensor, training, **kwargs):
        hidden_states = self.att_output(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        return self.LayerNorm(inputs=hidden_states + input_tensor)


class Intermediate(Layer):
    def __init__(self, model_opt: dict, activation, **kwargs):
        self.hidden_size = model_opt["HIDDEN_SIZE"]
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=model_opt["INIT_RANGE"])
        self.feedforward = Dense(
            units=self.hidden_size,
            kernel_initializer=self.initializer,
            activation=activation,
            name="FFNN"
        )
        super(Intermediate, self).__init__(**kwargs)

    def call(self, hidden_states):
        return self.feedforward(inputs=hidden_states)


class TransformerEncoder(Layer):
    def __init__(self, model_opt: dict, mid_activation, layer_num, **kwargs):
        self.config = model_opt
        self.attention_layer = SelfAttention(model_opt, name=f"ATT{i}")
        self.attrcln = ResConnLayerNorm(model_opt, name=f"RCLN{i}_0")
        self.intermediate = Intermediate(model_opt, mid_activation, name=f"FFNN{i}")
        self.interrcln = ResConnLayerNorm(model_opt, name=f"RCLN{i}_1")

        super(TransformerEncoder, self).__init__(**kwargs)

    def call(
            self,
            hidden_states: tf.Tensor,
            attention_mask: tf.Tensor,
            training: bool = False,
    ) -> tf.Tensor:
        input_states = hidden_states
        att_value = self.attention_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            training=training
        )
        mid_value = self.attrcln(
            hidden_states=att_value[0],
            input_tensor=input_states
        )
        intermediate_value = self.intermediate(mid_value)
        output = self.interrcln(
            hidden_states=intermediate_value,
            input_tensor=mid_value
        )

        return output
