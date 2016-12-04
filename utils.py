import h5py
import numpy as np
from vqaTools.vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import json


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


def getBatchData(data, split, BatchIds):
    features = data[split]['features']

    question_rep = []
    answer_rep = []
    img_rep = np.zeros((len(BatchIds), features.shape[1]))  # only consider one answer for each question

    images_batch = np.asarray(data[split]['images'])[BatchIds]
    for i, img in enumerate(images_batch):
        feat_id = img['feat_id']
        img_rep[i] = data[split]['features'][feat_id]
        question_rep.append(img['question'])
        answer_rep.append(img['ans'])

    return img_rep, question_rep, answer_rep


def evaluate_and_dump_predictions(pred, qids, qfile, afile, ix_ans_dict):
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
    fod_file = './eval/qa_predictions.json'
    fod = open(fod_file, 'wb')
    json.dump(answers, fod)
    fod.close()
    # VQA evaluation
    vqaRes = vqa.loadRes(fod_file, qfile)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    acc = vqaEval.accuracy['overall']
    print("Overall Accuracy is: %.02f\n" % acc)
    return acc
