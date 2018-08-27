import stat_utils as st 
from nltk.tokenize import word_tokenize
import numpy as np

def replace_unk_in_line(words, counts, thresh):
    """Helper function for replace_unk() for single line.
    
    Args:
        words (list of str): List of words forming a single sentence.
        counts (dict of (str, int)): Word counts for each word.
        thresh (int): Minimum number of words to not be '<UNK>'.

    Returns:
        list of str: List of words with low frequency replaced with <UNK>.
    """
    sentence = []

    for word in words:
        if counts[word] <= thresh:
            sentence.append('<UNK>')
        else:
            sentence.append(word)

    return sentence 
    

def replace_unk(inputs, outputs=[]):
    """Replaces uncommon words with '<UNK>' word.

    Args:
        inputs (list of list of str): Input list of tokenized sentences.
        outputs (list of list of str): Same as inputs.

    Returns: 
        list of list of str: List of tokenized sentences with
            low frequency words replaced with '<UNK>'.
        list of list of str: Same as inputs_u.
    """
    counts = st.word_counts(inputs+outputs)
    thresh = st.get_dict_thresh(counts)
    inputs_u = [replace_unk_in_line(line, counts, thresh) for line in inputs]
    outputs_u = [replace_unk_in_line(line, counts, thresh) for line in inputs]

    return inputs_u, outputs_u


def generate_dict(sentences):
    """Takes an array of strings and creates maps every unique word to 
    a unique integer.

    Args:
        sentences (list of list of str): Array of strings.

    Returns: 
        dict of (str, int): Mapping of word to unique integer (1-to-1).
    """
    flattened = [val for sublist in sentences for val in sublist]
    words = sorted(list(set(flattened)))
    word_dict = dict([(word, i) for i, word in enumerate(words)])
    return word_dict

def translate_data(inputs, outputs, dictionary):
    """Translates words to integers using unique mapping.

    Args:
        inputs (list of list of str): Tokenized and preprocessed sentences.
        outputs (list of list of str): Same as inputs.
        dictionary (dict of (str, int)): Mapping from word to unique integer.

    Returns:
        list of list of int: List of sentences encoded as integers for inputs.
        list of list of int: Same as above but for outputs.
    """
    return translate(inputs, dictionary), translate(outputs, dictionary)

def translate(sentences, dictionary):
    """Helper function for translate_data()"""
    translated = []

    for words in sentences:
        words_t = [dictionary[word] for word in words]
        translated.append(words_t)
    return translated

def one_hot_encode_target(outputs, vocab_size):
    """One hot encodes list of integer encode sentences as necessary for training.

    Args:
        outputs (list of list of int): Encoded output sentences.

    Returns:
        ndarray: 3D array representing one hot encoding of all sentences.
    """
    one_hot = np.zeros((len(outputs), len(outputs[0]), vocab_size), dtype='float32')

    for i, sentence in enumerate(outputs):
        for j, word in enumerate(outputs[0]):
            if j > 0:
                one_hot[i][j-1][word] = 1

    return one_hot

def pad_sentences(sentences,maxlen):
    """Adds '<PAD>' word to make sentence have length of maxlen to
    make all sentences the same length artificially.

    Args:
        sentences (list of list of str): List of tokenized sentences of 
            varying lengths.
        maxlen (int): User specified maximum length of sentence.

    Returns:
        list of list of str: List of tokenized sentences where each tokenized
            sentence contains multiple '<PAD>' characters to give them a length
            of maxlen.
    """
    sentences_u = []
    for sentence in sentences:
        padding = maxlen - len(sentence) + 1
        sentence.extend(['<PAD>']*padding)
        sentences_u.append(sentence)
    return sentences_u

def remove_long_sentences(inputs, outputs, maxlen):
    """Removes sentences with more than maxlen words.

    Args:
        inputs (list of list of str): List of tokenized sentences.
        outputs (list of list of str): List of tokenized sentences.
        maxlen (int): Maximum length of sentence to keep.

    Returns:
        list of list of str: List of tokenized sentences
            less than maxlen.
        list of list of str: Same as inputs_u
    """
    inputs_u = []
    outputs_u = []

    for i, o in zip(inputs, outputs):
        if len(i) <= 20 and len(o) <= 20:
            inputs_u.append(i)
            outputs_u.append(o)

    return inputs_u, outputs_u


def encode_sentences(inputs, outputs, maxlen):
    """Transforms list of sentence strings to list of of integer arrays.

    Args:
        inputs (list of str): List of raw input sentences.
        outputs (list of str): List of raw output sentences.
        maxlen (int): Max length of any sentence as user specified.

    Returns:
        list of list of int: List of int arrays where each int
            represents a word.
        list of list of int: List of int arrays where each int
            represents a word.
        dict of (int, string): Mapping of int to word for all
            words in vocab.
    """
    inputs = [word_tokenize(sentence.lower()) for sentence in inputs]
    outputs = [word_tokenize(sentence.lower()) for sentence in outputs]

    inputs, outputs = remove_long_sentences(inputs, outputs, maxlen)

    inputs, outputs = replace_unk(inputs, outputs)
    inputs = pad_sentences(inputs, maxlen)
    outputs = pad_sentences(outputs, maxlen)

    for i, sentence in enumerate(outputs):
        outputs[i] = ['<START>'] + sentence

    vocab = generate_dict(inputs+outputs)
    inputs, outputs = translate_data(inputs, outputs, vocab)

    vocab = dict((v,k) for k, v in vocab.items())
    return inputs, outputs, vocab



