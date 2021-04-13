import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

np.random.seed(42)
data_path = './data'
pos_corpus = 'positive.txt'
neg_corpus = 'negative.txt'

def load_dataset():
    pos_file = os.path.join(data_path, pos_corpus)
    neg_file = os.path.join(data_path, neg_corpus)

    pos_sents = []
    with open(pos_file, 'r') as f:
        for sent in f:
            pos_sents.append(sent.replace('\n', ''))

    neg_sents = []
    with open(neg_file, 'r') as f:
        for sent in f:
            neg_sents.append(sent.replace('\n', ''))

    balance_len = min(len(pos_sents), len(neg_sents))

    pos_df = pd.DataFrame(pos_sents, columns=['text'])
    pos_df['polarity'] = 1
    pos_df = pos_df[:balance_len]

    neg_df = pd.DataFrame(neg_sents, columns=['text'])
    neg_df['polarity'] = 0
    neg_df = neg_df[:balance_len]

    return pd.concat([pos_df, neg_df]).reset_index(drop=True)


print('Loading dataset...')
dataset = load_dataset()
dataset.to_csv('stock_comments_analyzed.csv', index=False)
print('Dataset size ', len(dataset))

X = dataset['text']
y = dataset['polarity'].astype(int)

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tokenizer.fit_on_texts(X)
vocab = tokenizer.word_index
print('Vocab size', len(vocab))

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
max_len = 64
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_word_ids = tokenizer.texts_to_sequences(X_train)
X_test_word_ids = tokenizer.texts_to_sequences(X_test)
X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=max_len)
X_test_padded_seqs = pad_sequences(X_test_word_ids, maxlen=max_len)
word_embedding = True

if word_embedding:
    print('Embedding...')
    EMBEDDING_FILE ='C:\\Users\dsw\\Desktop\\sgns.baidubaike.bigram-char\\sgns.baidubaike.bigram-char'
    embed_size = 300

    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf-8'))
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(vocab) + 1, embed_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

