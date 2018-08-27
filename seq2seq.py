from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.callbacks import ModelCheckpoint


batch_size = 64     # Number of training samples to use per step.
epochs = 100        # Total number of training epochs. 
latent_dim = 256    # Num of dimensions for word embedding space.

def generate_dataset(messages):

    return messages

def train_model(encoder_input_data, decoder_input_data, decoder_target_data):
    """Creates and trains encoder and decoder for seq2seq model.

    Args:
        encoder_input_data (list of list of int): Input sentences with words encoded
            as integers.
        decoder_input_data (list of list of int): Same as above but offset by 1 (has
            a '<START>' word at the beginning)
        decoder_target_data (ndarray): 3D one hot encoded object representing output sentences

    Returns:
        Model: encoder model for inference
        Model: decoder model for inference
    """
    num_encoder_tokens = len(decoder_target_data[0][0])
    num_decoder_tokens = num_encoder_tokens

    # Encoder 
    encoder_inputs = Input(shape=(None,))
    x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
    x, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(num_encoder_tokens, latent_dim)
    decoder_embedding_final = decoder_embedding(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding_final, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    filepath = "model/weight-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, 
              batch_size=batch_size, 
              epochs=epochs,
              callbacks=[checkpoint],
              validation_split=0.2)

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_embedding_inf = decoder_embedding(decoder_inputs)
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding_inf, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model, decoder_model

def save_model(filename, model):
    model.save(filename + '.h5')

def create_inference_model():
    return 

def generate_response(input_seq, encoder_model, decoder_model):
    #states_value = encoder_model.predict(input_seq)
    return

def train():
    # model = create_model()
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    # #do stuff
    return 