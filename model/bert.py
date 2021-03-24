import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Model

import transformer
import embedding


class BertEncoder(Layer):
    def __init__(self, model_opt, mid_activation, **kwargs):
        self.hidden_layer = model_opt["HIDDEN_LAYER_NUMS"]
        self.encoders = [transformer.TransformerEncoder(model_opt, mid_activation, name=f"Encoder_{i}")
                         for i in range(self.hidden_layer)]
        super(BertEncoder, self).__init__(**kwargs)

    def call(
            self,
            hidden_states: tf.Tensor,
            attention_mask: tf.Tensor,
            training: bool = False,
    ):
        for i, encoder in enumerate(self.encoders):
            outputs = encoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                training=training,
            )
            hidden_states = outputs
        return hidden_states


class Bert(Layer):
    def __init__(self, model_opt: dict, mid_activation, **kwargs):
        super(Bert, self).__init__(**kwargs)
        self.embedding = embedding.EmbeddingLayer(model_opt, name="Embedding")
        self.encoder = BertEncoder(model_opt, mid_activation, **kwargs)

    def call(
            self,
            input_ids: tf.Tensor,
            attention_mask: tf.Tensor,
            token_type_ids: tf.Tensor,
            training: bool = False,
    ):
        embedding_output = self.embedding.call(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        encoder_output = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            training=training,
        )
        return encoder_output

    def get_embedding_weight(self):
        return self.embedding.get_embedding_weight()
