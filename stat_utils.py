from collections import Counter

def word_counts(sentences):
    """Gets word counts given a list of sentences.

    Args:
        sentences (list of list of str): List of tokenized sentences.

    Returns:
        dict of (str, int): Mapping of all words in vocabulary to word count.
    """
    flatten = lambda l: [item for sublist in l for item in sublist]
    words = flatten(sentences)
    counts = Counter(words)
    return counts

def get_dict_thresh(counts):
    """Calculates minimum necessary word count to have 95% total 
    vocab coverage.

    Args:
        counts (dict of (str, int)): Word counts for all vocabulary.

    Returns:
        int: Minimum number of word count to be relevant.
    """
    total = sum(counts.values())
    current = 0 
    for val in sorted(counts.values(), reverse=True):
        current = current + val
        print("Val: " + str(val))
        print("Current: " + str(current/total))

        if current/total >= 0.94:
            return (val - 1)
    return -1


