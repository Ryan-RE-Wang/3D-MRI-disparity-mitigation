from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score, log_loss
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import skimage.transform as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
import re

def get_threshes(y_test, preds):
    tprs = []
    fprs = []
    thresholds = []

    for i in range(14):
        fpr, tpr, _ = roc_curve(y_test[:, i], preds[:, i], drop_intermediate=False)
        
        tprs.append(tpr)
        fprs.append(fpr)
        thresholds.append(_)
        
    return tprs, fprs, thresholds

def get_tpr(y_test, preds, thresh):
    tn, fp, fn, tp = confusion_matrix(y_test, np.where(preds >= thresh, 1, 0)).ravel()
    
    return tp/(tp+fn)

def get_fpr(y_test, preds, thresh):
    tn, fp, fn, tp = confusion_matrix(y_test, np.where(preds >= thresh, 1, 0)).ravel()
    
    return fp/(fp+tn)

def cal_best_thresh(y_test, y_preds):
    best_thresh = [0]* 14
    tprs, fprs, threshes = get_threshes(y_test, y_preds)
        
    for j in range(14):
        best = -1
        for i in range(len(threshes[j])):
            score = f1_score(y_test[:, j], np.where(y_preds[:, j] > threshes[j][i], 1, 0), average='binary')
            if (score > best):
                best = score
                best_thresh[j] = threshes[j][i]
                
    return best_thresh

def save_thresh(model, model_name, manager, X_val, y_val):
    
    model.load_weights(manager.checkpoints[0])
    
    if ('Adv' in model_name):
        y_preds = model.predict(X_val)[-1]
    else:
        y_preds = model.predict(X_val)

    best_thresh = cal_best_thresh(y_val, y_preds)

    np.savetxt('thresh/{i}_thresh.txt'.format(i=model_name), [best_thresh])
    

def expected_calibration_error(
    labels, pred_probs, num_bins=10):
    """
        Computes the calibration error with a binning estimator over equal sized bins
        See http://arxiv.org/abs/1706.04599 and https://arxiv.org/abs/1904.01685.
        Does not currently support sample weights
    """

    bin_ids = pd.qcut(pred_probs, num_bins, labels=False, retbins=False, duplicates='drop')
    
    df = pd.DataFrame({"pred_probs": pred_probs, "labels": labels, "bin_id": bin_ids})
    ece_df = (
        df.groupby("bin_id")
        .agg(
            pred_probs_mean=("pred_probs", "mean"),
            labels_mean=("labels", "mean"),
            bin_size=("pred_probs", "size"),
        )
        .assign(
            bin_weight=lambda x: x.bin_size / df.shape[0],
            err=lambda x: np.abs(x.pred_probs_mean - x.labels_mean),
        )
    )
    
    result = np.average(ece_df.err.values, weights=ece_df.bin_weight)
    
    return result

def cal_auc(y_test, preds):
    auc_score = []

    for i in range(y_test.shape[1]):
        fpr, tpr, _ = roc_curve(y_test[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        
        auc_score.append(roc_auc)
        
    return auc_score


def test(y_preds, y_test, best_thresh):
    
    n_bootstraps = 1000
    rng_seed = 2021  # control reproducibility
    result_df = pd.DataFrame()

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        
        indices = rng.randint(0, len(y_preds), len(y_preds))
        
        if len(np.unique(y_test[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        
        auc = roc_auc_score(y_test[indices], y_preds[indices])
    
        bce_score = log_loss(y_test[indices], y_preds[indices])

        ece_score = expected_calibration_error(y_test[indices], y_preds[indices])

        y_preds_ = np.where(y_preds > best_thresh, 1, 0)

        tn, fp, fn, tp = confusion_matrix(y_test[indices], y_preds_[indices]).ravel()
        er = (fp + fn) / (tn + fp + fn + tp)
        pr = (fp + tp) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        
        result_df_dict = pd.DataFrame({"AUC":auc, "BCE":bce_score, "ECE":ece_score, "Positive rate":pr, "Error rate":er, "Precision":precision}, index=[i])
        
        result_df = pd.concat([result_df, result_df_dict])
    
    return result_df

def demo_test(y_preds, y_test):
    
    n_bootstraps = 1000
    rng_seed = 2021  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_preds), len(y_preds))
        
        if len(np.unique(y_test[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

                
        score = roc_auc_score(y_test[indices], y_preds[indices])
        bootstrapped_scores.append(score)
        
    
    auc_score = np.array(bootstrapped_scores)
    
    mean_score = auc_score.mean()
    std_dev = auc_score.std()
    std_error = std_dev / np.math.sqrt(1)
    ci =  2.262 * std_error
    lower_bound = mean_score - ci
    upper_bound = mean_score + ci

    print("Sample auc mean: {:0.3f}". format(mean_score))
    print("Samole auc std: {:0.3f}".format(std_dev))
    print("Sample auc CI: {:0.3f}". format(ci))
    print("Confidence interval for the score: [{:0.3f} - {:0.3f}]".format(
        lower_bound, upper_bound))


def legend_without_duplicate_labels(figure):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(), loc='upper right')