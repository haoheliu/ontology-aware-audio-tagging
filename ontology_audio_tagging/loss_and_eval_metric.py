import numpy as np
from tqdm import tqdm
import os
import warnings
import sklearn
import torch
from numba import jit
from ontology_audio_tagging.utils import load_pickle, save_pickle
from ontology_audio_tagging.audioset_graph_distance import get_audioset_graph_distance
import torch.nn as nn

GRAPH_WEIGHT = None


def ontology_binary_cross_entropy(output, labels):
    epsilon = 1e-7
    output = torch.clamp(output, epsilon, 1.0 - epsilon)
    loss = nn.BCELoss(reduction="none")(output, labels)
    loss_weight = ontology_loss_weight(labels)
    loss = (torch.mean(loss * loss_weight) + torch.mean(loss)) / 2
    return loss


def ontology_loss_weight(target, beta=1):
    # Target: [132, 527]
    # GRAPH_WEIGHT: [527, 527]
    global GRAPH_WEIGHT

    if GRAPH_WEIGHT is None:
        GRAPH_WEIGHT = torch.tensor(
            get_audioset_graph_distance(), requires_grad=False
        ).float()
        GRAPH_WEIGHT = GRAPH_WEIGHT / torch.max(GRAPH_WEIGHT)

    GRAPH_WEIGHT = GRAPH_WEIGHT.to(target.device)
    graph_weight = GRAPH_WEIGHT**beta

    weight = []
    for i in range(target.shape[0]):
        res = target[i : i + 1] * graph_weight
        res[res == 0] = torch.inf  # res==0 means the element is not in the target
        weight.append(torch.min(res, dim=1)[0].unsqueeze(0))
    weight = torch.cat(weight, dim=0)

    # If the target only have one class, the weight on that class will be inf
    weight[weight == torch.inf] = 0.0

    # Normalize the weight value
    weight = weight / torch.max(weight, dim=1, keepdim=True)[0]
    weight[target > 0] = 1.0
    weight = weight / torch.mean(weight)
    return weight


def precision_recall_curve(
    y_true, probas_pred, *, pos_label=None, tps_weight=None, fps_weight=None
):
    fps, tps, thresholds = _binary_clf_curve(
        y_true,
        probas_pred,
        pos_label=pos_label,
        tps_weight=tps_weight,
        fps_weight=fps_weight,
    )

    ps = tps + fps
    precision = np.divide(tps, ps, where=(ps != 0))

    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]

    # reverse the outputs so recall is decreasing
    sl = slice(None, None, -1)
    return np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0)), thresholds[sl]


def _binary_clf_curve(
    y_true, y_score, pos_label=None, tps_weight=None, fps_weight=None
):
    # Weight is the false positive re-weighting for each sample (18k+)
    # Original label: speech; Pred label: speech, conversation, male speech; What is the false positive weight when we calculate the class 'conversation'?

    # Check to make sure y_true is valid
    y_type = sklearn.utils.multiclass.type_of_target(y_true)  # , input_name="y_true"
    if not (y_type == "binary" or (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    sklearn.utils.check_consistent_length(y_true, y_score, fps_weight)
    sklearn.utils.check_consistent_length(y_true, y_score, tps_weight)
    y_true = sklearn.utils.column_or_1d(y_true)
    y_score = sklearn.utils.column_or_1d(y_score)
    sklearn.utils.assert_all_finite(y_true)
    sklearn.utils.assert_all_finite(y_score)

    # Filter out zero-weighted samples, as they should not impact the result
    if fps_weight is not None:
        fps_weight = sklearn.utils.column_or_1d(fps_weight)
        fps_weight = sklearn.utils.validation._check_sample_weight(fps_weight, y_true)
        # nonzero_weight_mask = sample_weight != 0
        # y_true = y_true[nonzero_weight_mask]
        # y_score = y_score[nonzero_weight_mask]
        # sample_weight = sample_weight[nonzero_weight_mask]

    # Filter out zero-weighted samples, as they should not impact the result
    if tps_weight is not None:
        tps_weight = sklearn.utils.column_or_1d(tps_weight)
        tps_weight = sklearn.utils.validation._check_sample_weight(tps_weight, y_true)
        # nonzero_weight_mask = sample_weight != 0
        # y_true = y_true[nonzero_weight_mask]
        # y_score = y_score[nonzero_weight_mask]
        # sample_weight = sample_weight[nonzero_weight_mask]
    pos_label = sklearn.metrics._ranking._check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    if tps_weight is None:
        y_true = y_true == pos_label  # y_true stand for if the sample is positive
    else:
        # y_true[y_true==0] = tps_weight[y_true==0]
        y_true = tps_weight
        assert np.max(y_true) <= 1.0

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    # array([9.8200458e-01, 9.7931880e-01, 9.7723687e-01, ..., 8.8192965e-04, 7.5673062e-04, 5.9041742e-04], dtype=float32)
    y_score = y_score[desc_score_indices]
    # array([ True,  True,  True, ..., False, False, False])
    y_true = y_true[desc_score_indices]

    if fps_weight is not None:
        weight = fps_weight[desc_score_indices]
    else:
        weight = 1.0
    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    # tps = sklearn.utils.extmath.stable_cumsum(y_true * weight)[threshold_idxs]
    tps = sklearn.utils.extmath.stable_cumsum(y_true)[threshold_idxs]

    if fps_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = sklearn.utils.extmath.stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def _average_precision(y_true, pred_scores, tps_weight=None, fps_weight=None):
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, pred_scores, tps_weight=tps_weight, fps_weight=fps_weight
    )
    precisions = numpy.array(precisions)
    recalls = numpy.array(recalls)
    AP = numpy.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    return AP


def initialize_weight(graph_weight_path):
    # print("Normalize graph connectivity weight by the max value.")
    weight = np.load(graph_weight_path)
    return weight


def mask_weight(weight, threshold=1.0):
    # ones_matrix = np.ones_like(weight)
    ones_matrix = weight.copy()
    ones_matrix[weight <= threshold] *= 0
    diag = np.eye(ones_matrix.shape[0]) == 1
    if np.mean(ones_matrix[~diag]) > 1e-9:
        ones_matrix = ones_matrix / np.mean(ones_matrix[~diag])
    return ones_matrix


@jit(nopython=True)
def build_ontology_fps_sample_weight_min(target, weight, class_idx):
    ret = []
    for i in range(target.shape[0]):
        positive_indices = np.where(target[i] == 1)[0]
        assert np.sum(positive_indices) > 0, (
            "the %d-th sample in the evaluation set do not have positive label" % i
        )
        minimum_distance_with_class_idx = np.min(weight[positive_indices][:, class_idx])
        ret.append(minimum_distance_with_class_idx)
    return ret


def build_weight(target, weight, refresh=False):
    ret = {}
    path = "ontology_weight.pkl.tmp"
    if refresh or not os.path.exists(path):
        print("Build ontology based metric weight")
        for threshold in tqdm(
            np.linspace(0, int(np.max(weight)), int(np.max(weight)) + 1)
        ):
            ret[threshold] = {}
            masked = mask_weight(weight, threshold)
            for i in range(target.shape[1]):
                ret[threshold][i] = build_ontology_fps_sample_weight_min(
                    target, masked, i
                )
        save_pickle(ret, path)
    else:
        ret = load_pickle(path)
    return ret


def ontology_mean_average_precision(target, clipwise_output):
    graph_node_distance = get_audioset_graph_distance()
    ontology_weight = build_weight(target, graph_node_distance)

    omap_on_different_coarse_level_details = {}

    while True:
        try:
            for threshold in tqdm(
                np.linspace(
                    0,
                    int(np.max(graph_node_distance)),
                    int(np.max(graph_node_distance)) + 1,
                )
            ):
                fps_ap = []
                for i in range(target.shape[1]):
                    fps_weight = ontology_weight[threshold][i]
                    fps_ap.append(
                        _average_precision(
                            target[:, i],
                            clipwise_output[:, i],
                            tps_weight=None,
                            fps_weight=fps_weight,
                        )
                    )
                omap_on_different_coarse_level_details[threshold] = np.array(fps_ap)
            break
        except:
            ontology_weight = build_weight(target, graph_node_distance, refresh=True)

    omap_on_different_coarse_level = {
        k: np.mean(v) for k, v in omap_on_different_coarse_level_details.items()
    }
    omap_average = np.mean([v for k, v in omap_on_different_coarse_level.items()])

    return (
        omap_average,
        omap_on_different_coarse_level,
        omap_on_different_coarse_level_details,
    )
