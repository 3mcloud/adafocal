'''
Metrics to measure calibration of a trained deep neural network.
This is directly taken from the publicly available code at https://github.com/torrvision/focal_calibration/blob/main/Metrics/metrics.py
'''
import collections

import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Some keys used for the following dictionaries
COUNT = 'count'
CONF = 'conf'
CORRECT_CONF = 'correct_conf'
INCORRECT_CONF = 'incorrect_conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


def _bin_initializer(bin_dict, num_bins):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][CORRECT_CONF] = 0
        bin_dict[i][INCORRECT_CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0

def _populate_bins(confs, preds, labels, num_bins):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]

        if math.isnan(num_bins * confidence):
            print(confs)
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        # if confidence == 0.0, then binn = -1
        if binn == -1: binn=0

        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][CORRECT_CONF] = bin_dict[binn][CORRECT_CONF] + (confidence if (label == prediction) else 0)
        bin_dict[binn][INCORRECT_CONF] = bin_dict[binn][INCORRECT_CONF] + (0 if (label == prediction) else confidence)
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(bin_dict[binn][COUNT])
    return bin_dict


def expected_calibration_error(confs, preds, labels, num_bins):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    bin_stats_dict = collections.defaultdict(dict)
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)

        # Save the bin stats to be returned
        bin_lower, bin_upper = i*(1/num_bins), (i+1)*(1/num_bins)
        bin_stats_dict[i]['lower_bound'] = bin_lower
        bin_stats_dict[i]['upper_bound'] = bin_upper
        bin_stats_dict[i]['prop_in_bin'] = float(bin_count)/num_samples
        bin_stats_dict[i]['accuracy_in_bin'] = bin_accuracy
        bin_stats_dict[i]['avg_confidence_in_bin'] = bin_confidence
        bin_stats_dict[i]['ece'] = bin_stats_dict[i]['avg_confidence_in_bin'] - bin_stats_dict[i]['accuracy_in_bin']

    return ece, bin_stats_dict


def maximum_calibration_error(confs, preds, labels, num_bins):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    ce = []
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        ce.append(abs(bin_accuracy - bin_confidence))
    return max(ce)


# Mukhoti et. al. implementation at https://github.com/torrvision/focal_calibration/blob/main/Metrics/metrics.py
def adaECE_error_mukhoti(confs, preds, labels, num_bins=15):
    def histedges_equalN(x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, num_bins + 1), np.arange(npt), np.sort(x))
        # y = numpy.interp(x, xp, fp) returns the interpolated function values for pints in x using xp-fp points already given. 
        # Let x = np.array([0.26, 0.53, 0.61, 0.75, 0.94, 0.99])
        # np.linspace(0, 6, 3 + 1) -> array([ 0.,  2.,  4., 6.])
        # np.arange(6) -> array([0, 1, 2, 3, 4, 5])
        # np.interp(np.linspace(0, 6, 3+1), np.arange(6), np.sort(x)) -> array([0.26, 0.61, 0.94, 0.99])

    confidences, predictions, labels = torch.FloatTensor(confs), torch.FloatTensor(preds), torch.FloatTensor(labels)
    accuracies = predictions.eq(labels)
    n, bin_boundaries = np.histogram(confidences.cpu().detach(), histedges_equalN(confidences.cpu().detach()))

    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = torch.zeros(1)
    bin_dict = collections.defaultdict(dict)
    bin_num = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            accuracy_in_bin = torch.zeros(1)
            avg_confidence_in_bin = torch.zeros(1)
        # Save the bin stats to be returned
        bin_dict[bin_num]['lower_bound'] = bin_lower
        bin_dict[bin_num]['upper_bound'] = bin_upper
        bin_dict[bin_num]['prop_in_bin'] = prop_in_bin.item()
        bin_dict[bin_num]['accuracy_in_bin'] = accuracy_in_bin.item()
        bin_dict[bin_num]['avg_confidence_in_bin'] = avg_confidence_in_bin.item()
        bin_dict[bin_num]['calibration_gap'] = bin_dict[bin_num]['avg_confidence_in_bin'] - bin_dict[bin_num]['accuracy_in_bin']
        bin_num += 1
        
    return ece, bin_dict


def test_classification_net_adafocal(model, data_loader, device, num_bins, num_labels=None):
    '''
    This function reports classification accuracy and confusion matrix over a dataset.
    '''
    model.eval()
    labels_list = []
    predictions_list = []
    confidence_vals_list = []
    loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(data_loader):
            data = data.to(device)
            labels = labels.to(device)
            
            logits = model(data)
            # Compute NLL (cross entropy loss)
            loss += F.cross_entropy(logits, labels, reduction='sum').item()

            if i == 0:
                fulldataset_logits = logits
            else:
                fulldataset_logits = torch.cat((fulldataset_logits, logits), dim=0)

            log_softmax = F.log_softmax(logits, dim=1)
            log_confidence_vals, predictions = torch.max(log_softmax, dim=1)
            confidence_vals = log_confidence_vals.exp()

            predictions_list.extend(predictions.cpu().numpy().tolist())
            confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())
            labels_list.extend(labels.cpu().numpy().tolist())

            num_samples += len(data)

    accuracy = accuracy_score(labels_list, predictions_list)
    return loss/num_samples, confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
        predictions_list, confidence_vals_list, fulldataset_logits



class ClasswiseECELoss(nn.Module):
    '''
    Compute Classwise ECE
    '''
    def __init__(self, n_bins):
        super(ClasswiseECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce
