from keras.models import Sequential
from keras.layers.core import Dense, Merge, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Embedding
from keras.layers.normalization import BatchNormalization


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
        activation = 'tanh'
        dropout = 0.5
        vocab_dim = 200
        vocab_size = self.vocab['q_vocab_size']
        output_classes = self.vocab['a_vocab_size']

        image_model = Sequential()
        image_model.add(BatchNormalization(mode=0, input_shape=(self.feat_size, )))
        image_model.add(Dense(hidden_units, input_dim=self.feat_size, activation=activation))
        image_model.add(Dropout(dropout))

        language_model = Sequential()
        language_model.add(Embedding(output_dim=vocab_dim, input_dim=vocab_size + 2, mask_zero=True,
                                     init='uniform'))
        language_model.add(LSTM(hidden_units, input_shape=(max_len, vocab_size), return_sequences=False))
        language_model.add(Dropout(dropout))

        model = Sequential()
        model.add(Merge([language_model, image_model], mode='mul', concat_axis=1))
        for i in xrange(num_hidden_layers):
            model.add(Dense(hidden_units, init='uniform', activation=activation))
            model.add(Dropout(dropout))
        model.add(Dense(output_classes, activation='softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
