import os
import fasttext

data_dir = os.environ['SM_CHANNEL_TRAINING']

name_corpus = 'corpus.txt'
path_corpus = os.path.join(data_dir, name_corpus)

model_type = 'skipgram'
learning_rate = 0.05
dim_vectors = 150

model = fasttext.train_unsupervised(path_corpus, model=model_type, lr=learning_rate, dim=dim_vectors)

word = 'gato'
word_vector = model.get_word_vector(word)
similar_words = model.get_nearest_neighbors(word)

print(word)
print(word_vector)
print(similar_words)