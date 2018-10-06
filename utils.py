import numpy as np
import re
import pandas
import itertools
from collections import Counter
from sklearn.utils import shuffle


def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def rem_htmltags(s):
    s = re.sub(r'<[^>]+>','',s)
    s = re.sub(r'&(nbsp;)',' ',s)
    s = re.sub(r'<[^>]+','',s)
    return s


def load_data_from_disk():
    # Load dataset from file
    df_ham = pandas.read_table("data/ham.csv", sep=",",encoding='latin-1', low_memory=False)
    df_spam = pandas.read_table("data/spam.csv", sep=",",encoding='latin-1',low_memory=False)

    # df_ham = pandas.read_table(CORPUS_DIR + "/ham.csv", sep=",", low_memory=False)
    # df_spam = pandas.read_table(CORPUS_DIR + "/spam.csv", sep=",", low_memory=False)

    # remove all Unnamed Columns form the CSV File
    df_ham.drop(list(df_ham.filter(regex='Unnamed')), axis=1, inplace=True)
    df_spam.drop(list(df_spam.filter(regex='Unnamed')), axis=1, inplace=True)

    # concatenate both SUBJECT and BODY
    df_ham['message'] = df_ham.SUBJECT.str.cat(df_ham.BODY)
    df_spam['message'] = df_spam.SUBJECT.str.cat(df_spam.BODY)

    # drop the columns SUBJECT from both ham and spam files
    df_ham.drop(['SUBJECT', 'BODY'], axis=1, inplace=True)
    df_spam.drop(['SUBJECT', 'BODY'], axis=1, inplace=True)

    # adding labels
    df_ham['label'] = 'ham'
    df_spam['label'] = 'spam'

    # merge both data frames HAM and SPAM into One.
    df = df_ham.append(df_spam, ignore_index=True)
    df = shuffle(df)

    # very important otherwise df[0]->(message) length and df[1]->(label) length are mismatched
    df = df[pandas.notnull(df['message'])]
    # drop all NaN rows from the data frame
    df.dropna()



    # Split by words
    X = [clean_str(sentence) for sentence in df['message']]
    X = [list(sentence) for sentence in X]
    Y = [[0, 1] if (label == 'spam') else [1, 0] for label in df['label']]

    return [X, Y]


def pad_sentences(sentences, padding_word="<PAD/>", maxlen=0):
    """
    Pads all the sentences to the same length. The length is defined by the longest sentence.
     Returns padded sentences.
    """

    if maxlen > 0:
        sequence_length = maxlen
    else:
        sequence_length = max(len(s) for s in sentences)

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        replaced_newline_sentence = []
        for char in list(sentence):
            if char == "\n":
                replaced_newline_sentence.append("<NEWLINE/>")
            elif char == " ":
                replaced_newline_sentence.append("<SPACE/>")
            else:
                replaced_newline_sentence.append(char)

        new_sentence = replaced_newline_sentence + [padding_word] * num_padding

        # new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """

    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))

    # Map from index to word
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Map from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary
    """
    x = np.array([[vocabulary[word] if word in vocabulary else 0 for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def sentence_to_index(sentence, vocabulary, maxlen):
    sentence = clean_str(sentence)
    raw_input = [list(sentence)]
    sentences_padded = pad_sentences(raw_input, maxlen=maxlen)
    raw_x, dummy_y = build_input_data(sentences_padded, [0], vocabulary)
    return raw_x


def load_data():
    sentences, labels = load_data_from_disk()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    return [x, y, vocabulary, vocabulary_inv]


if __name__ == '__main__' :
    load_data()