import tensorflow as tf


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    def input_fn(params):
        """The actual input function."""
        batch_size = params["BATCH_SIZE"]

        name_to_features = {
            "input_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
                tf.io.FixedLenFeature([1], tf.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))
            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))
            d = d.interleave(
                tf.data.TFRecordDataset,
                cycle_length=cycle_length
            )
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.map(lambda record: _decode_record(record, name_to_features),
                  num_parallel_calls=num_cpu_threads)
        d = d.batch(batch_size=batch_size, drop_remainder=True)
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


def build_interleaved_tfrecord_dataset(
        tfrecord_paths: list,
        max_sequence_length: int,
        max_predictions_per_seq: int,
        batch_size: int,
        num_cpu_threads: int
):
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecord_paths))
    dataset = dataset.shuffle(buffer_size=len(tfrecord_paths))
    dataset = dataset.repeat()

    cycle_length = min(num_cpu_threads, len(tfrecord_paths))

    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=cycle_length,
        block_length=num_cpu_threads * 4,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.shuffle(buffer_size=10)

    name_to_features = {
        "input_ids":
            tf.io.FixedLenFeature([max_sequence_length], tf.int64),
        "input_mask":
            tf.io.FixedLenFeature([max_sequence_length], tf.int64),
        "segment_ids":
            tf.io.FixedLenFeature([max_sequence_length], tf.int64),
        "masked_lm_positions":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.io.FixedLenFeature([1], tf.int64),
    }
    dataset = dataset.map(lambda record: _decode_record(record, name_to_features),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
