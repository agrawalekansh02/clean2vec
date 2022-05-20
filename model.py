import tensorflow as tf

class Seq2Seq:
    def __init__(self, embedding_shape=50, vocab_size=5500, hidden_size=256, dropout_rate=0.5):
        # encoder layer
        self.encoder_inputs = tf.keras.layers.Input(shape=(None,))
        self.encoder_embedding = tf.keras.layers.Embedding(vocab_size, 
            embedding_shape, 
            mask_zero=True)(self.encoder_inputs)
        self.encoder_output, self.encoder_hidden, self.encoder_cell = tf.keras.layers.LSTM(hidden_size, 
            return_state=True, 
            dropout=dropout_rate)(self.encoder_embedding)
        self.encoder_states = [self.encoder_hidden, self.encoder_cell]

        # decoder layer
        self.decoder_inputs = tf.keras.layers.Input(shape=(None,))
        self.decoder_embedding = tf.keras.layers.Embedding(vocab_size, 
            embedding_shape, 
            mask_zero=True)(self.decoder_inputs)
        self.decoder_lstm = tf.keras.layers.LSTM(hidden_size, 
            return_state=True, 
            return_sequences=True,
            dropout=dropout_rate)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_embedding,  
                                initial_state=self.encoder_states)

        # final dense layer
        self.decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
        self.output = self.decoder_dense(self.decoder_outputs)

        self.model = tf.keras.Model(inputs=[self.encoder_inputs, self.decoder_inputs], outputs=self.output)

    def inference(self, embedding_shape=50):
        # inputs for vectors based on encoder
        decoder_state_hidden = tf.keras.layer.Input(shape=(embedding_shape,))
        decoder_state_cell = tf.keras.layer.Input(shape=(embedding_shape,))
        decoder_states_inputs = [decoder_state_hidden, decoder_state_cell]

        # adding embedded layer as third input
        decoder_outputs, state_hidden, state_cell = self.decoder_lstm(self.decoder_embedding,
                                    initial_state=decoder_states_inputs)
        decoder_states = [state_hidden, state_cell]

        # final dense layer to softmax singular word
        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = tf.keras.Model(inputs=[self.decoder_inputs]+decoder_states_inputs, 
            outputs=[decoder_outputs]+decoder_states)

        # inital prompt as first input for encoder
        encoder_model = tf.keras.Model(inputs=self.encoder_inputs,
            outputs=self.encoder_states)

        return decoder_model, encoder_model

    