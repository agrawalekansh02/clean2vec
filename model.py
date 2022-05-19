import tensorflow as tf

class Seq2Seq:
    def __init__(self, input_shape=(27,), output_shape=50, vocab_size=5500, hidden_size=256, dropout_rate=0.5):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    # creates embedding layer, encoders, and decoders
    def build(self):
        # embeding
        # embedding_layer = tf.keras.layers.Embedding(self.vocab_size, 
        #     output_dim=self.output_shape, 
        #     input_length = self.input_shape,
        #     trainable=True,
        #     name='embedding_layer')

        # encoder layers
        encoder_input = tf.keras.layers.Input(shape=self.input_shape)
        # encoder_embedding = embedding_layer(encoder_input)
        encoder_gru_layer1 = tf.keras.layers.LSTM(self.hidden_size,
            return_state=True, 
            dropout=self.dropout_rate, 
            name='encoder_gru_1')
        _, hidden, cell = encoder_gru_layer1(encoder_input)
        encoder_states = [hidden, cell]

        # decoder layers        
        decoder_input = tf.keras.layers.Input(shape=self.input_shape)
        # decoder_embedding = embedding_layer(decoder_input)
        decoder_gru_layer1 = tf.keras.layers.LSTM(self.hidden_size, 
            return_sequences=True, 
            return_state=True,
            dropout=self.dropout_rate,  
            name='decoder_gru_1')
        decoder_output, _, _ = decoder_gru_layer1(encoder_input, 
            initial_state=encoder_states)
        dense_output = tf.keras.layers.Dense(self.vocab_size, 
            activation='softmax', 
            name='decoder_dense')(decoder_output)

        model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=dense_output)
        return model
