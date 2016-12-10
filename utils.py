import h5py
from vqaTools.vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import json
import os


def load_questions_answers(filePath):
    with h5py.File(filePath, 'r') as hf:
        questions_train = hf.get(u'ques_train').value
        questions_test = hf.get(u'ques_test').value
        ans_train = hf.get(u'answers').value
    return questions_train, questions_test, ans_train


def load_positions_ids(filePath):
    with h5py.File(filePath, 'r') as hf:
        img_train_pos = hf.get(u'img_pos_train').value
        img_val_pos = hf.get(u'img_pos_test').value
        q_test_id = hf.get(u'question_id_test').value
    return img_train_pos, img_val_pos, q_test_id


def evaluate_and_dump_predictions(pred, qids, qfile, afile, ix_ans_dict, filename):
    """
    dumps predictions to some default file
    :param pred: list of predictions, like [1, 2, 3, 2, ...]. one number for each example
    :param qids: question ids in the same order of predictions, they need to align and match
    :param qfile:
    :param afile:
    :param ix_ans_dict:
    :return:
    """
    assert len(pred) == len(qids), "Number of predictions need to match number of question IDs"
    answers = []
    for i, val in enumerate(pred):
        qa_pair = {}
        qa_pair['question_id'] = int(qids[i])
        qa_pair['answer'] = ix_ans_dict[str(val + 1)]  # note indexing diff between python and torch
        answers.append(qa_pair)
    vqa = VQA(afile, qfile)
    fod = open(filename, 'wb')
    json.dump(answers, fod)
    fod.close()
    # VQA evaluation
    vqaRes = vqa.loadRes(filename, qfile)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    acc = vqaEval.accuracy['overall']
    print("Overall Accuracy is: %.02f\n" % acc)
    return acc


def determine_filename(filename, extension=''):
    keep_iterating = True
    count = 0
    while keep_iterating:
        # making sure to not save the weights as the same as an existing one
        count += 1
        unused_name = filename + str(count) + extension
        if not os.path.isfile(unused_name):
            keep_iterating = False
    return unused_name