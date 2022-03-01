import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


cpdef double calc_accuracy(int[::1] labels_ground_truth, int[::1] labels_predicted):
    """
    Compute the accuracy by comparing the ground truth labels with the predicted ones
    
    :param labels_ground_truth: 
    :param labels_predicted: 
    :return: double accuracy
    """
    correctly_classified = np.sum(np.array(labels_predicted) == np.array(labels_ground_truth))
    accuracy = 100 * (correctly_classified / len(labels_ground_truth))

    return accuracy

cpdef double calc_f1(int[::1] labels_ground_truth, int[::1] labels_predicted, str name_dataset):
    np_gt = np.array(labels_ground_truth)
    np_pred = np.array(labels_predicted)

    if name_dataset == 'enzymes':
        return 0

    # print(f'precision: {precision_score(np_gt, np_pred):.2f}')
    # print(f'recall: {recall_score(np_gt, np_pred):.2f}')
    return f1_score(np.array(labels_ground_truth), np.array(labels_predicted))

cpdef double calc_save_accuracy(int[::1] labels_ground_truth, int[::1] labels_predicted,
                                str name, str folder):
    """
    Compute the accuracy and save the predicted labels
    
    :param labels_ground_truth: 
    :param labels_predicted: 
    :param name: 
    :param folder: 
    :return: double accuracy
    """
    filename = os.path.join(folder, name)
    with open(filename, 'wb') as f:
        np.save(f, np.array(labels_ground_truth))
        np.save(f, np.array(labels_predicted))

    accuracy = calc_accuracy(labels_ground_truth, labels_predicted)

    return accuracy
