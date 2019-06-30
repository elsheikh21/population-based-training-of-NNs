import keras
from keras import Input, Model
from keras.layers import concatenate, dot, Embedding, GRU, Dense, Activation
from hyperparameters import DropoutHP


def define_nmt(hidden_size, embedding_size, timesteps, src_vocab_size, tar_vocab_size,
               dropout, lr):
    # layers with parameters
    encoder_emb = Embedding(src_vocab_size, embedding_size)
    encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True)
    decoder_emb = Embedding(tar_vocab_size, embedding_size)
    decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True)
    decoder_tan = Dense(hidden_size, activation="tanh")
    decoder_drop = DropoutHP(rate=dropout)
    decoder_softmax = Dense(tar_vocab_size, activation='softmax')

    def define_encoder(encoder_inputs):
        encoder_embed = encoder_emb(encoder_inputs)
        encoder_out, encoder_state = encoder_gru(encoder_embed)
        return encoder_out, encoder_state

    def define_decoder(decoder_inputs, encoder_states, decoder_init_state):
        decoder_embed = decoder_emb(decoder_inputs)
        decoder_out, decoder_state = decoder_gru(decoder_embed, initial_state=decoder_init_state)
        attention = dot([decoder_out, encoder_states], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder_states], axes=[2, 1])
        decoder_context = concatenate([context, decoder_out])
        decoder_pred = decoder_tan(decoder_context)
        decoder_pred = decoder_drop(decoder_pred)
        decoder_pred = decoder_softmax(decoder_pred)
        return decoder_pred, decoder_state

    # joint model for training
    encoder_inputs = Input(shape=(timesteps,))
    decoder_inputs = Input(shape=(timesteps - 1,))
    encoder_out, encoder_state = define_encoder(encoder_inputs)
    decoder_pred, _ = define_decoder(decoder_inputs, encoder_out, encoder_state)
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    # encoder inference model
    encoder_inf_inputs = Input(shape=(timesteps,))
    encoder_inf_out, encoder_inf_state = define_encoder(encoder_inf_inputs)
    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])

    # decoder inference model
    encoder_inf_states = Input(shape=(timesteps, hidden_size,))
    decoder_init_state = Input(shape=(hidden_size,))
    decoder_inf_inputs = Input(shape=(1,))
    decoder_inf_pred, decoder_inf_state = define_decoder(decoder_inf_inputs, encoder_inf_states, decoder_init_state)
    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, decoder_inf_state])

    return model, encoder_model, decoder_model

'''
class NMT(keras.Model):

    def __init__(self, hidden_size, embedding_size, timesteps, src_vocab_size, tar_vocab_size, dropout):
        super().__init__(name='nmt')
        self.timesteps = timesteps
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size

        self.encoder_emb = Embedding(src_vocab_size, embedding_size, input_shape=(timesteps,))
        self.encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True)
        self.decoder_emb = Embedding(tar_vocab_size, embedding_size, input_shape=(timesteps - 1,))
        self.decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True)
        self.decoder_tan = Dense(hidden_size, activation="tanh")
        self.decoder_drop = DropoutHP(rate=dropout)
        self.decoder_softmax = Dense(tar_vocab_size, activation='softmax')

    def load_weights(self, filepath, **kwargs):
        # hack to force model initialization
        t = self.timesteps
        self.test_on_batch([np.zeros((1, t)), np.zeros((1, t - 1))],
                           keras.utils.to_categorical(np.zeros((1, t - 1)), self.tar_vocab_size))
        super().load_weights(filepath)

    def call(self, inputs, training):
        encoder_inputs, decoder_inputs = inputs
        encoder_out, encoder_state = self.encode(encoder_inputs)
        decoder_pred, _ = self.decode(decoder_inputs, encoder_out, encoder_state, training)
        return decoder_pred

    def encode(self, encoder_inputs):
        encoder_embed = self.encoder_emb(encoder_inputs)
        encoder_out, encoder_state = self.encoder_gru(encoder_embed)
        return encoder_out, encoder_state

    def decode(self,  decoder_inputs, encoder_states, decoder_init_state, training):
        decoder_embed = self.decoder_emb(decoder_inputs)
        decoder_out, decoder_state = self.decoder_gru(decoder_embed, initial_state=decoder_init_state)
        attention = dot([decoder_out, encoder_states], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder_states], axes=[2, 1])
        decoder_context = concatenate([context, decoder_out])
        decoder_pred = self.decoder_tan(decoder_context)
        if training:
            decoder_pred = self.decoder_drop(decoder_pred)
        decoder_pred = self.decoder_softmax(decoder_pred)
        return decoder_pred, decoder_state

    def infer(self, inputs):
        preds = np.full((inputs.shape[0], inputs.shape[1] - 1), Tokenizer.PAD)
        decoder_inputs = np.full(inputs.shape[0], Tokenizer.BOS)

        encoder_out, encoder_state = self.encode(inputs)
        decoder_state = encoder_state

        index = np.arange(len(inputs))
        for t in range(inputs.shape[1]):
            decoder_pred, decoder_state = \
                self.decode(encoder_out[index], decoder_state, np.expand_dims(decoder_inputs, axis=1))
            decoder_max_pred = np.argmax(decoder_pred, axis=-1)[:, 0]

            next_index = decoder_max_pred != Tokenizer.PAD
            if not any(next_index):
                break

            index = index[next_index]
            decoder_state = decoder_state[next_index]
            decoder_inputs = decoder_max_pred[next_index]
            preds[t, index] = decoder_inputs

        return preds.T
'''