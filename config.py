options = {}

# global settings
options['dataset_root'] ='/home/ted/research/data/' # put all your files here
options['qah5'] = 'data_prepro_noshuffled_LSTMCNN.h5' # this file contains question features, image positions, etc.
options['img_train'] = 'vqa_img_vgg_fc_trainval_noshuffled.h5' # image features file
options['test_annfile'] = 'mscoco_val2014_annotations.json' # annotations for the validation set
options['test_questionfile'] = 'OpenEnded_mscoco_val2014_questions.json' # questions for validation set
options['vocab_img_datafile'] = 'vqa_data_prepro.json' # contains unique image ids, word to index mapping, answer to index mapping

# training parameters
options['max_epochs']=100
options['patience']= 5
options['batch_size']=128