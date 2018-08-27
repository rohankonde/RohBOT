import numpy as np
from nltk.tokenize import word_tokenize

class Rohbot:
    encoder = None
    decoder = None
    vocab = None
    reverse_vocab = None

    def __init__(self, encoder, decoder, vocab):
        self.encoder = encoder
        self.decoder = decoder
        self.reverse_vocab = vocab

        self.vocab = dict()
        for k, v in vocab.items():
            self.vocab[v] = k


    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1,1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = 479
    # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder.predict(
                [target_seq] + states_value)
    # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_vocab[sampled_token_index]
            decoded_sentence += ' '+sampled_char
    # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '<PAD>' or
               len(decoded_sentence) > 20):
                stop_condition = True
    # Update the target sequence (of length 1).
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index
    # Update states
            states_value = [h, c]
        return decoded_sentence

    def encode_sentence(self, sentence):
        sentence = word_tokenize(sentence.lower())
        sentence = [self.vocab[x] for x in sentence]
        return sentence

    def talk(self, sentence):
        return self.decode_sequence(self.encode_sentence(sentence))