################################################
## Helper functies voor het embedding model   ##
## Floris Menninga                            ##  
## Datum: 09-01-2026                          ##  
################################################


import io
import re
import string
import tqdm

import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow as tf
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE
SEED = 0


class EmbeddingModel:

    def __init__(self, sequence_length=1000, vocab_size=25000, embedding_dim=16):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.text_ds = None
        self.vectorize_layer = None
        self.sequences = None
        self.dataset = None
        self.w2v_model = None

    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase,
                                        '[%s]' % re.escape(string.punctuation), '')

    def load_dataset(self, path="/home/floris/Downloads/trainingsData.txt"):
        self.text_ds = tf.data.TextLineDataset([path]).filter(
            lambda x: tf.cast(tf.strings.length(x), bool)
        )

    def create_vocab(self):
        self.vectorize_layer = layers.TextVectorization(
            standardize=self.custom_standardization,
            max_tokens=self.vocab_size,
            output_mode='int',
            output_sequence_length=self.sequence_length
        )
        
        self.vectorize_layer.adapt(self.text_ds.batch(512))


    def vectorize(self):
        text_vector_ds = self.text_ds.batch(512).prefetch(AUTOTUNE).map(self.vectorize_layer).unbatch()
        self.sequences = list(text_vector_ds.as_numpy_iterator())

    def _custom_skipgrams(sequence, vocabulary_size, window_size=4, sampling_table=None):
        couples = []
        for i, wi in enumerate(sequence):
            if not wi:
                continue
            if sampling_table is not None:
                if sampling_table[wi] < random.random():
                    continue

            window_start = max(0, i - window_size)
            window_end = min(len(sequence), i + window_size + 1)
            for j in range(window_start, window_end):
                if j != i:
                    wj = sequence[j]
                    couples.append([wi, wj])
        return couples

    def _generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
        targets, contexts, labels = [], [], []
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

        for sequence in tqdm.tqdm(sequences):
            positive_skip_grams = EmbeddingModel._custom_skipgrams(
                sequence,
                vocabulary_size=vocab_size,
                sampling_table=sampling_table,
                window_size=window_size
            )

            for target_word, context_word in positive_skip_grams:
                context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
                
                negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=num_ns,
                    unique=True,
                    range_max=vocab_size,
                    seed=seed,
                    name="negative_sampling")

                context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
                
                label = tf.constant([1] + [0] * num_ns, dtype="float32")

                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

        return targets, contexts, labels

    def generate_trainingdata(self, window_size=5, num_ns=3, batch_size=256, buffer_size=10000):
        targets, contexts, labels = self._generate_training_data(
            sequences=self.sequences,
            window_size=window_size,
            num_ns=num_ns,
            vocab_size=self.vocab_size,
            seed=SEED)

        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        self.dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    def word2vec(self):
        class Word2Vec(tf.keras.Model):
            def __init__(self, vocab_size, embedding_dim):
                super(Word2Vec, self).__init__()
                self.target_embedding = layers.Embedding(vocab_size,
                                                  embedding_dim,
                                                  name="w2v_embedding")
                self.context_embedding = layers.Embedding(vocab_size,
                                                   embedding_dim)

            def call(self, pair):
                target, context = pair
                if len(target.shape) == 2:
                    target = tf.squeeze(target, axis=1)
                word_emb = self.target_embedding(target)
                context_emb = self.context_embedding(context)
                dots = tf.einsum('be,bce->bc', word_emb, context_emb)
                return dots

        self.w2v_model = Word2Vec(self.vocab_size, self.embedding_dim)


    def loss_function(self, y_true, y_pred):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)

    def fit(self, epochs=13, log_dir="logs"):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        self.w2v_model.compile(optimizer='adam',loss=self.loss_function, metrics=['accuracy'])

        self.w2v_model.fit(self.dataset, epochs=epochs, callbacks=[tensorboard_callback])

    def save2file(self, vector_file='vectors.tsv', meta_file='metadata.tsv'):
        weights = self.w2v_model.get_layer('w2v_embedding').get_weights()[0]
        vocab = self.vectorize_layer.get_vocabulary()

        num_tokens = min(len(vocab), len(weights))

        with io.open(vector_file, 'w', encoding='utf-8') as out_v, \
             io.open(meta_file, 'w', encoding='utf-8') as out_m:

            for index in range(1, num_tokens):
                word = vocab[index]
                vec = weights[index]
                
                clean_word = word.strip()
                if not clean_word:
                    clean_word = "[UNK_EMPTY]"

                out_v.write('\t'.join([str(x) for x in vec]) + "\n")
                out_m.write(clean_word + "\n")

        print(f"Saved {num_tokens - 1} vectors and metadata entries.")


# Test:

# model = EmbeddingModel()
# model.load_dataset()
# model.create_vocab()
# model.vectorize()
# model.generate_trainingdata()
# model.word2vec()
# model.fit()
# model.save2file()