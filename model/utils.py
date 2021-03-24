import tensorflow as tf
#
# def input_preprocss(dataset):
#     processed = {
#         "input_ids" = dataset,
#     }
#
# def parse_function(example_proto, name_to_features):
#     example = tf.io.parse_single_example(example_proto, name_to_features)
#     for name in list(example.keys()):
#         t = example[name]
#         if t.dtype == tf.int64:
#             t = tf.cast(t, tf.int32)
#         if name == "attention_mask":
#             t = tf.cast(t, tf.float32)
#         example[name] = t
#     return example
#
#
# def build_interleaved_tfrecord_dataset(
#     tfrecord_paths: List[str], max_sequence_length: int, batch_size: int, num_cpu_threads: int
# ):
#     dataset = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecord_paths))
#     dataset = dataset.shuffle(buffer_size=len(tfrecord_paths))
#     dataset = dataset.repeat()
#
#     cycle_length = min(num_cpu_threads, len(tfrecord_paths))
#
#     dataset = dataset.interleave(
#         tf.data.TFRecordDataset,
#         cycle_length=cycle_length,
#         block_length=num_cpu_threads * 4,
#         num_parallel_calls=tf.data.experimental.AUTOTUNE,
#     )
#     dataset = dataset.shuffle(buffer_size=10)
#
#     name_to_features = {
#         "input_ids": tf.io.FixedLenFeature([max_sequence_length], tf.int64),
#         "attention_mask": tf.io.FixedLenFeature([max_sequence_length], tf.float32),
#         "segment_ids": tf.io.FixedLenFeature([max_sequence_length], tf.int64),
#         "label": tf.io.FixedLenFeature([], tf.int64),
#     }
#
#     feature_parse_fn = partial(parse_function, name_to_features=name_to_features)
#     dataset = dataset.map(feature_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     return dataset
#
# class TFBertPreTrainingLoss:
#     """
#     Loss function suitable for BERT-like pretraining, that is, the task of pretraining a language model by combining
#     NSP + MLM. .. note:: Any label of -100 will be ignored (along with the corresponding logits) in the loss
#     computation.
#     """
#
#     def compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
#         loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
#             from_logits=True, reduction=tf.keras.losses.Reduction.NONE
#         )
#         # make sure only labels that are not equal to -100
#         # are taken into account as loss
#         masked_lm_active_loss = tf.not_equal(tf.reshape(tensor=labels["labels"], shape=(-1,)), -100)
#         masked_lm_reduced_logits = tf.boolean_mask(
#             tensor=tf.reshape(tensor=logits[0], shape=(-1, shape_list(logits[0])[2])),
#             mask=masked_lm_active_loss,
#         )
#         masked_lm_labels = tf.boolean_mask(
#             tensor=tf.reshape(tensor=labels["labels"], shape=(-1,)), mask=masked_lm_active_loss
#         )
#         next_sentence_active_loss = tf.not_equal(tf.reshape(tensor=labels["next_sentence_label"], shape=(-1,)), -100)
#         next_sentence_reduced_logits = tf.boolean_mask(
#             tensor=tf.reshape(tensor=logits[1], shape=(-1, 2)), mask=next_sentence_active_loss
#         )
#         next_sentence_label = tf.boolean_mask(
#             tensor=tf.reshape(tensor=labels["next_sentence_label"], shape=(-1,)), mask=next_sentence_active_loss
#         )
#         masked_lm_loss = loss_fn(y_true=masked_lm_labels, y_pred=masked_lm_reduced_logits)
#         next_sentence_loss = loss_fn(y_true=next_sentence_label, y_pred=next_sentence_reduced_logits)
#         masked_lm_loss = tf.reshape(tensor=masked_lm_loss, shape=(-1, shape_list(next_sentence_loss)[0]))
#         masked_lm_loss = tf.reduce_mean(input_tensor=masked_lm_loss, axis=0)
#
#         return masked_lm_loss + next_sentence_loss
#

def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    batch_size = sequence_tensor.shape[0]
    seq_length = sequence_tensor.shape[1]
    width = sequence_tensor.shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor
