import utils
import preproc as pp
import pickle
from stat_utils import word_counts, get_dict_thresh
from seq2seq import train_model, save_model
import numpy as np

inputs, outputs = utils.load_sequences('data/inputs.pkl', 'data/outputs.pkl')

#inputs = inputs[:100]
#outputs = outputs[:100]

#inputs, outputs, vocab = pp.encode_sentences(inputs, outputs, 20)

inputs = utils.load_dict('data/inputs_translated.pkl')
outputs = utils.load_dict('data/outputs_translated.pkl')
vocab = utils.load_dict('data/vocabulary.pkl')

# with open('data/vocabulary.pkl', 'wb') as handle:
# 	pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

one_hot_target = pp.one_hot_encode_target(outputs, len(vocab))

encoder, decoder = train_model(np.array(inputs), np.array(outputs), one_hot_target)
save_model('encoder_inference_100', encoder)
save_model('decoder_inference_100', decoder)


# save_model('model/encoder_0.01', encoder)
# save_model('model/decoder_0.01', decoder)