import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import json
import logging
import torch.nn.functional as F
import os
import numpy as np
from utils import to_cuda, to_numpy
# from torch.optim.lr_scheduler import StepLR
from csls import CSLS
from collections import OrderedDict
from data import WordDictionary, MonoDictionary, Language,\
    CrossLingualDictionary


class Evaluator(object):
    """
        Computes different metrics for evaluation purposes.
        Currently supported evaluation metrics
            - Unsupervised CSLS metric for model selection
            - Cross-Lingual precision matches
    """

    def __init__(self, src_lang, tgt_lang, data_dir):
        # Load Cross-Lingual Data
        """
            :param src_lang (Language): The source Language
            :param tgt_lang (Language): The Target Language
            :param data_dir (str): The data directory
                (assumes cross-lingual dictionaries are kept in
                data_dir/crosslingual/{src}-{tgt}.txt)
        """
        assert (isinstance(src_lang, Language) and
                isinstance(tgt_lang, Language) and
                isinstance(data_dir, str))
        self.data_dir = data_dir
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self.logger = logging.getLogger()

    def supervised(
        self, csls, metrics,
        mode="csls", word_dict=None
    ):
        """
        Reports the precision at k (1, 5, 10) accuracy for
        word translation using facebook dictionaries
        """
        if not hasattr(self, 'word_dict'):
            cross_lingual_dir = os.path.join(
                self.data_dir, "crosslingual", "dictionaries")
            cross_lingual_file = "%s-%s.5000-6500.txt" % (
                self.src_lang.name, self.tgt_lang.name)
            cross_lingual_file = os.path.join(
                cross_lingual_dir, cross_lingual_file)
            self.word_dict = WordDictionary(
                self.src_lang, self.tgt_lang, cross_lingual_file)
        word_dict = word_dict or self.word_dict
        predictions, _ = self.get_match_samples(
            csls, word_dict.word_map[:, 0], 10, mode=mode, use_mean=False)
        _metrics, total = word_dict.precisionatk(predictions, [1, 5, 10])
        metrics['total'] = total
        metrics['acc1'] = _metrics[0]
        metrics['acc5'] = _metrics[1]
        metrics['acc10'] = _metrics[2]
        total = metrics["total"]
        acc1 = metrics["acc1"]
        acc5 = metrics["acc5"]
        acc10 = metrics["acc10"]
        self.logger.info(
            f"Total: {total:d}, "
            f"Precision@1: {acc1:5.2f}, @5: {acc5:5.2f}, @10: {acc10:5.2f}"
        )
        return metrics

    def unsupervised(self, csls, metrics, mode="csls"):
        max_src_word_considered = 10000
        _, metrics['unsupervised'] = self.get_match_samples(
            csls, np.arange(
                min(self.src_lang.vocab, int(max_src_word_considered))),
            1, mode=mode)
        self.logger.info(
            "{0:12s}: {1:5.4f}".format(
                "Unsupervised", metrics['unsupervised']))
        return metrics

    def monolingual(self, csls, metrics, **kwargs):
        if not hasattr(self, 'mono_dict'):
            mono_lingual_dir = os.path.join(self.data_dir, "monolingual")
            self.mono_dict = MonoDictionary(self.src_lang, mono_lingual_dir)
        if not self.mono_dict.atleast_one:
            return metrics
        metrics = self.mono_dict.get_spearman_r(csls, metrics)
        self.logger.info("=" * 64)
        self.logger.info("{0:>25s}\t{1:5s}\t{2:10s}\t{3:5s}".format(
            "Dataset", "Found", "Not Found", "Corr"))
        self.logger.info("=" * 64)
        mean = -1
        for dname in metrics['monolingual']:
            if dname == "mean":
                mean = metrics['monolingual'][dname]
                continue
            found = metrics['monolingual'][dname]['found']
            not_found = metrics['monolingual'][dname]['not_found']
            correlation = metrics['monolingual'][dname]['correlation']
            self.logger.info(
                "{0:>25s}\t{1:5d}\t{2:10d}\t{3:5.4f}".format(
                    dname, found, not_found, correlation))
        self.logger.info("=" * 64)
        self.logger.info("Mean Correlation: {0:.4f}".format(mean))
        return metrics

    def crosslingual(self, csls, metrics, **kwargs):
        if not hasattr(self, 'cross_dict'):
            cross_lingual_dir = os.path.join(self.data_dir, "crosslingual")
            self.cross_dict = CrossLingualDictionary(
                self.src_lang, self.tgt_lang, cross_lingual_dir)
        if not self.cross_dict.atleast_one:
            return metrics
        metrics = self.cross_dict.get_spearman_r(csls, metrics)
        self.logger.info("=" * 64)
        self.logger.info(
            "{0:>25s}\t{1:5s}\t{2:10s}\t{3:5s}".format(
                "Dataset", "Found", "Not Found", "Corr"))
        self.logger.info("=" * 64)
        mean = -1
        for dname in metrics['crosslingual']:
            if dname == "mean":
                mean = metrics['crosslingual'][dname]
                continue
            found = metrics['crosslingual'][dname]['found']
            not_found = metrics['crosslingual'][dname]['not_found']
            correlation = metrics['crosslingual'][dname]['correlation']
            self.logger.info(
                "{0:>25s}\t{1:5d}\t{2:10d}\t{3:5.4f}".format(
                    dname, found, not_found, correlation))
        self.logger.info("=" * 64)
        self.logger.info("Mean Correlation: {0:.4f}".format(mean))

    def evaluate(self, csls, evals, mode="csls"):
        """
        Evaluates the csls object on the functions specified in evals
            :param csls: CSLS object, which contains methods to
                         map source space to target space and such
            :param evals: list(str): The functions to evaluate on
            :return metrics: dict: The metrics computed by evaluate.
        """
        metrics = {}
        for eval_func in evals:
            assert hasattr(self, eval_func), \
                "Eval Function {0} not found".format(eval_func)
            metrics = getattr(self, eval_func)(csls, metrics, mode=mode)
        return metrics

    def get_match_samples(self, csls, range_indices, n,
                          use_mean=True, mode="csls", hubness_thresh=20):
        """
        Computes the n nearest neighbors for range_indices (from src)
        wrt the target in csls object.
        For use_mean True, this computes the avg metric for all
        top k nbrs across batch, and reports the average top 1 metric.
            :param csls : The csls object
            :param range_indices: The source words (from 0, ... )
            :param n: Number of nbrs to find
            :param use_mean (bool): Compute the mean or not
        """
        target_indices, metric = csls.get_closest_csls_matches(
            source_indices=range_indices,
            n=n, mode=mode)
        if use_mean:
            logger = logging.getLogger(__name__)
            # Hubness removal
            unique_tgts, tgt_freq = np.unique(target_indices,
                                              return_counts=True)
            hubs = unique_tgts[tgt_freq > hubness_thresh]
            old_sz = metric.shape[0]
            filtered_metric = metric[np.isin(
                target_indices, hubs, invert=True)]
            logger.info(
                "Removed {0:d} hub elements For unsupervised".format(
                    old_sz - filtered_metric.shape[0]))
            mean_metric = np.mean(filtered_metric)
            return target_indices, mean_metric
        else:
            return target_indices, metric