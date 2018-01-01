import 
from keras.models import Model
from keras.layers import Input, LSTM, Dense

num_encoder_tokens = 10
def generate_dataset(messages):

	return dataset

def create_model():
	encoder_inputs = Input(shape=(None,))
	x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
	x, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
	encoder_states = [state_h, state_c]

	decoder_inputs = Input(shape=(None,))
	x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
	x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
	decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	return model

def create_inference_model():
	return inf_model

def generate_response():
	return response 

def train():
	model = create_model()
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
	model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.1)
	#do stuff
