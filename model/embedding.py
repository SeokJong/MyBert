import tensorflow as tf
from tensorflow.keras.layers import Layer


class EmbeddingLayer(Layer):
    def __init__(self, config: dict, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.vocab_size = config["VOCAB_SIZE"]
        self.type_vocab_size = config["VOCAB_TOKEN"]
        self.hidden_size = config["HIDDEN_SIZE"]
        self.max_position_embeddings = config["MAX_POS_EMBED"]
        self.embeddings_sum = tf.keras.layers.Add()
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config["NORM_EPS"], name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config["DROPOUT_RATE"])
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=config["INIT_RANGE"])
        with tf.name_scope("word"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=self.initializer,
            )
        with tf.name_scope("segment"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.type_vocab_size, self.hidden_size],
                initializer=self.initializer,
            )

        with tf.name_scope("position"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=self.initializer,
            )


    def call(
            self,
            input_ids: tf.Tensor = None,
            token_type_ids: tf.Tensor = None,
            training: bool = False,
    ) -> tf.Tensor:
        inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = inputs_embeds.shape[:-1]
        position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))

        if token_type_ids == None:
            token_type_ids = tf.zeros(input_ids.shape, dtype=tf.int32)
        print(token_type_ids)

        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)

        final_embeddings = self.embeddings_sum(inputs=[inputs_embeds, position_embeds, token_type_embeds])
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
        return final_embeddings

    def get_embedding_weight(self):
        return self.weight
