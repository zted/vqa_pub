import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Embedding


class KerasModel(object):
    """ model initialization """

    def __init__(self, feat_size, vocab, options):
        self.feat_size = feat_size
        self.vocab = vocab
        self.options = options

    """ build model """

    def build_model(self, max_len):

        num_hidden_layers = 2
        hidden_units = 1024
        activation = 'relu'
        dropout = 0.5
        vocab_size = self.vocab['q_vocab_size']
        output_classes = self.vocab['a_vocab_size']
        image_model = Sequential()
        image_model.add(Dense(32, input_dim=self.feat_size))
        vocab_dim = 300
        embedding_weights = np.random.rand(vocab_size + 2, vocab_dim)

        language_model = Sequential()
        language_model.add(Embedding(output_dim=vocab_dim, input_dim=vocab_size + 2, mask_zero=True,
                                     weights=[embedding_weights]))
        language_model.add(LSTM(128, input_shape=(max_len, vocab_size)))
        language_model.add(Dropout(0.5))

        model = Sequential()
        model.add(Merge([language_model, image_model], mode='concat', concat_axis=1))
        for i in xrange(num_hidden_layers):
            model.add(Dense(hidden_units, init='uniform'))
            model.add(Activation(activation))
            model.add(Dropout(dropout))
        model.add(Dense(output_classes))
        model.add(Activation('softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
