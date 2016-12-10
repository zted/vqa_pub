import json
import time

import h5py
import numpy as np
from keras.utils.generic_utils import Progbar
from keras.utils.np_utils import to_categorical

import utils as u
from config import *
from keras_models import KerasModel

dataset_root = options['dataset_root']
qah5_path = dataset_root + options['qah5']
img_path = dataset_root + options['img_train']

# -------------For evaluation on validation data--------------------
annFile = dataset_root + options['test_annfile']
quesFile = dataset_root + options['test_questionfile']

#------------------Vocabulary indices--------------------------------
vocab_img_data_path = dataset_root + options['vocab_img_datafile']

#------------------Loading image positions and question IDs-----------------
img_train_pos, img_val_pos, q_test_id = u.load_positions_ids(qah5_path)
q_train, q_val, a_train = u.load_questions_answers(qah5_path)
a_train = to_categorical(a_train-1, a_train.max())
# the - 1 is for offset. so class #1 would be 0, it must be done because to_categorical starts at 0

q_maxlen = len(q_train[0])

# Find the file names that we want to save weights and evaluation to. Make a file
# there so that other programs that just begin to run won't use the same name
tmpweights = './tmp/weights'
tmpweights = u.determine_filename(tmpweights, '.hdf5')
f = open(tmpweights, 'w')
f.close()

evaldump = './eval/qa_predictions'
evaldump = u.determine_filename(evaldump, '.json')
f = open(evaldump, 'w')
f.close()

print("Save weights to {}\nDump predictions to {}".format(tmpweights, evaldump))

# prepare data
print 'Reading %s' % (vocab_img_data_path,)
data = json.load(open(vocab_img_data_path, 'r'))
with h5py.File(img_path, 'r') as hf:
    img_train = hf.get(u'images_train').value
    img_val = hf.get(u'images_test').value
img_feature_size = len(img_val[0])

print("Shapes:\nImage Train - {}\nImage Val - {}\nQ Train - {}\nQ Val - {}\nAns Train - {}"
      .format(img_train.shape, img_val.shape, q_train.shape, q_val.shape, a_train.shape))

vocab = {}
vocab['ix_to_word'] = data['ix_to_word']
vocab['q_vocab_size'] = len(vocab['ix_to_word'])
vocab['ix_to_ans'] = data['ix_to_ans']
vocab['a_vocab_size'] = len(vocab['ix_to_ans'])

# --------------------Training Parameters--------------------
batch_size = options.get('batch_size', 128)
nb_epoch = options.get('max_epochs', 100)
shuffle = options.get('shuffle', True)
max_patience = options.get('patience', 5)
batch_size = options['batch_size']

index_array = np.arange(len(q_train))

print('Building model...')
mod = KerasModel(img_feature_size, vocab, options)
model = mod.build_model(q_maxlen)
print('Train...')

best_yet = 0
patience = 0
total_time = time.time()
for e in range(nb_epoch):
    print("Training epoch {}".format(e + 1))
    pbar = Progbar(1 + len(q_train) / batch_size)
    np.random.shuffle(index_array)
    start_time = time.time()

    nb_batch = int(np.ceil(len(index_array) / float(batch_size)))
    train_acc = 0.0
    for batch_index in range(0, nb_batch):
        batch_start = batch_index * batch_size
        batch_end = min(len(index_array), (batch_index + 1) * batch_size)
        current_batch_size = batch_end - batch_start
        training_indices = index_array[batch_start:batch_end]

        q_train_batch = np.array([q_train[i] for i in training_indices])
        i_pos_batch = [img_train_pos[i] for i in training_indices]
        i_train_batch = np.array([img_train[i-1] for i in i_pos_batch])
        # i - 1 because positions were recorded as starting from 1
        a_batch = np.array([a_train[i] for i in training_indices])

        history = model.fit([q_train_batch, i_train_batch], a_batch, batch_size=current_batch_size, nb_epoch=1, verbose=False)
        train_acc += history.history['acc'][-1]
        # because we're manually doing batch training but still want accuracy,
        # we tally up the accuracies from each batch and average them later.
        pbar.update(batch_index)

    print("\nFinished training epoch {}. Accuracy = {:.3f}".format(e+1, train_acc*100/(batch_size*nb_batch)))
    # the accuracy should be slightly higher, because we are rounding up the number of examples when
    # we multiply batch size * nb_batch
    print("Time taken to train epoch: {}".format(int(time.time() - start_time)))

    if (e + 1) % options['epochs_to_validate'] == 0:
        # we validate every few epochs
        i_val_batch = np.array([img_val[i - 1] for i in img_val_pos])
        pred = model.predict_classes([q_val, i_val_batch], batch_size=100)
        val_acc = u.evaluate_and_dump_predictions(pred, q_test_id, quesFile, annFile, vocab['ix_to_ans'], evaldump)
        print("Validation Accuracy: {:.3f}\n".format(val_acc))

        if val_acc > best_yet:
            print('Accuracy improved from {} to {}, saving weights to {}'.format(best_yet, val_acc, tmpweights))
            best_yet = val_acc
            model.save_weights(tmpweights, overwrite=True)
            patience = 0
        else:
            patience += 1

    if patience > max_patience:
        print('Out of patience. No improvement after {} epochs'.format(patience*options['epochs_to_validate']))
        break

print("Best accuracy achieved was {}".format(best_yet))
print("Total time taken was {} seconds".format(time.time() - total_time))
