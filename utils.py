from fbparser import MessageHistory, MessageThread
import pickle

def getSequencePairs(thread, target='Rohan Kondetimmanahalli'):
    prev_sender = None
    prev_message = None

    inputs = []
    outputs = []

    for sender, message in zip(thread.senders, thread.messages):
        if prev_sender != target and prev_sender is not None and sender == target:
            inputs.append(prev_message)
            outputs.append(message)
        prev_sender = sender
        prev_message = message

    return inputs, outputs

def generate_dataset(history):
    inputs = []
    outputs = []

    for thread in history.threads:
        thread_clean = thread.crunch().clean()
        input_part, output_part = getSequencePairs(thread_clean)
        inputs.extend(input_part)
        outputs.extend(output_part)

    return inputs, outputs

def save_sequences(inputs, outputs):
    with open('data/inputs.pkl', 'wb') as f:
        pickle.dump(inputs, f)

    with open('data/outputs.pkl', 'wb') as f:
        pickle.dump(outputs, f)

    return

def load_sequences(inputPath, outputPath):
    with open(inputPath, 'rb') as f:
        inputs = pickle.load(f)

    with open(outputPath, 'rb') as f:
        outputs = pickle.load(f)

    return inputs, outputs

def save_dict(dictionary, fname):
    with open(fname, 'wb') as f:
        pickle.dump(dictionary, f)

def load_dict(fname):
    with open(fname, 'rb') as f:
        dictionary = pickle.load(f)
    return dictionary





