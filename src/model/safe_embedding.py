import tensorflow as tf

class SAFEEmbedding(tf.keras.models.Model):
    def __init__(self, u=5, e=200, n=64, attention_depth=25, attention_hops=10, c=0.5, *args, **kwargs):
        kwargs["name"] = kwargs.get("name", "SAFE_embedding")
        super().__init__(*args, **kwargs)
        self.u = 2 * u
        self.e = e
        self.n = n
        self.d_a = attention_depth
        self.r = attention_hops
        self.c = c
        self.bidirectional_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(u, return_sequences=True))
        self.flatten = tf.keras.layers.Flatten()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "u": self.u // 2,
                "e": self.e,
                "n": self.n,
                "attention_depth": self.d_a,
                "attention_hops": self.r,
                "c": self.c
            }
        )
        return config

    def build(self, input_shape):
        self.ws1 = self.add_weight(name="W_s1",
                                   shape=(self.d_a, self.u),
                                   initializer="random_normal",
                                   trainable=True)

        self.ws2 = self.add_weight(name="W_s2",
                                   shape=(self.r, self.d_a),
                                   initializer="random_normal",
                                   trainable=True)

        self.wout1 = self.add_weight(name="W_out1",
                                     shape=(self.e, (self.r * self.u)),
                                     initializer="random_normal",
                                     trainable=True)

        self.wout2 = self.add_weight(name="W_out2",
                                     shape=(self.n, self.e),
                                     initializer="random_normal",
                                     trainable=True)

    def call(self, inputs, training=None, mask=None):
        H = self.bidirectional_rnn(inputs, mask=mask)

        A = tf.matmul(self.ws1, H, transpose_b=True)
        A = tf.tanh(A)
        A = tf.matmul(self.ws2, A)
        A = tf.nn.softmax(A)

        B = tf.matmul(A, H)
        C = self.flatten(B)

        C = tf.expand_dims(C, axis=2)
        f1 = tf.matmul(self.wout1, C)
        f1 = tf.nn.relu(f1)
        f = tf.matmul(self.wout2, f1)
        f = tf.squeeze(f, axis=2)

        return f

    def regularization_term(self, attention_matrix):
        AAT = tf.matmul(attention_matrix, attention_matrix, transpose_b=True)
        I = tf.eye(num_rows=self.r, num_columns=self.r)
        B = tf.subtract(AAT, I)
        C = tf.norm(B, ord='fro', axis=[1, 2])
        D = tf.reduce_mean(C)
        return D
