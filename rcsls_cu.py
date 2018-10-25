#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import argparse
import sys
from rutils_cu import *
from utils import to_numpy, to_cuda
import torch
import time

parser = argparse.ArgumentParser(description='RCSLS for supervised word alignment')

parser.add_argument("--src_emb", type=str, default='', help="Load source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Load target embeddings")
parser.add_argument('--center', action='store_true', help='whether to center embeddings or not')

parser.add_argument("--dico_train", type=str, default='', help="train dictionary")
parser.add_argument("--dico_test", type=str, default='', help="validation dictionary")

parser.add_argument("--output", type=str, default='', help="where to save aligned embeddings")

parser.add_argument("--knn", type=int, default=10, help="number of nearest neighbors in RCSL/CSLS")
parser.add_argument("--maxneg", type=int, default=200000, help="Maximum number of negatives for the Extended RCSLS")
parser.add_argument("--maxsup", type=int, default=-1, help="Maximum number of training examples")
parser.add_argument("--maxload", type=int, default=200000, help="Maximum number of loaded vectors")

parser.add_argument("--model", type=str, default="none", help="Set of constraints: spectral or none")
parser.add_argument("--reg", type=float, default=0.0 , help='regularization parameters')

parser.add_argument("--lr", type=float, default=1.0, help='learning rate')
parser.add_argument("--niter", type=int, default=10, help='number of iterations')
parser.add_argument('--sgd', action='store_true', help='use sgd')
parser.add_argument("--batchsize", type=int, default=10000, help="batch size for sgd")
parser.add_argument("--gpuid", type=int, help="Set to GPU number >= 0")

params = parser.parse_args()

gpuid = params.gpuid
print("Using GPU ", gpuid)

###### SPECIFIC FUNCTIONS ######
# functions specific to RCSLS
# the rest of the functions are in utils.py

def getknn(sc, x, y, k=10):
    sidx = torch.topk(sc, k, dim=1)[1]
    ytopk = y[sidx.view(-1), :]
    ytopk = ytopk.reshape(sidx.shape[0], sidx.shape[1], y.shape[1])
    f = torch.sum(sc[np.arange(sc.shape[0])[:, None], sidx])
    df = torch.mm(ytopk.sum(1).t(), x)
    return f / k, df / k

def rcsls(X_src, Y_tgt, Z_src, Z_tgt, R, knn=10):
    X_trans = torch.mm(X_src, R.t())
    f = 2 * torch.sum(X_trans * Y_tgt)
    df = 2 * torch.mm(Y_tgt.t(), X_src)
    fk0, dfk0 = getknn(torch.mm(X_trans, Z_tgt.t()), 
                       X_src,
                       Z_tgt,
                       knn)

    obj1 = torch.mm(Z_src, R.t())
    obj2 = Y_tgt.t()

    # CUDA memory error when multiplying these 200000x300 and 300x200000
    #   so converting to np
    # obj_fin = torch.mm(obj1, obj2).t() 
    obj1 = to_numpy(obj1, gpuid>=0)
    obj2 = to_numpy(obj2, gpuid>=0)
    obj_fin = np.dot(obj1, obj2).T
    obj_fin = to_cuda(torch.Tensor(obj_fin), gpuid)

    fk1, dfk1 = getknn(obj_fin, Y_tgt, Z_src)

    f = f - fk0 - fk1
    df = df - dfk0 - dfk1.t()

    return -f / X_src.shape[0], -df / X_src.shape[0]


def proj_spectral(R):
    R_np = to_numpy(R, gpuid>=0)
    U, s, V = np.linalg.svd(R_np)
    s[s > 1] = 1
    s[s < 0] = 0
    return np.dot(U, np.dot(np.diag(s), V))


###### MAIN ######

# load word embeddings
words_tgt, x_tgt_np = load_vectors(params.tgt_emb, maxload=params.maxload, center=params.center)
words_src, x_src_np = load_vectors(params.src_emb, maxload=params.maxload, center=params.center)

# load validation bilingual lexicon
src2tgt, lexicon_size = load_lexicon(params.dico_test, words_src, words_tgt)

# word --> vector indices
idx_src = idx(words_src)
idx_tgt = idx(words_tgt)


# load train bilingual lexicon
pairs = load_pairs(params.dico_train, idx_src, idx_tgt)
if params.maxsup > 0 and params.maxsup < len(pairs):
    pairs = pairs[:params.maxsup]

x_src = to_cuda(torch.Tensor(x_src_np), gpuid)
x_tgt = to_cuda(torch.Tensor(x_tgt_np), gpuid)

# selecting training vector  pairs
X_src_train, Y_tgt_train = select_vectors_from_pairs(x_src, x_tgt, pairs, gpuid=gpuid)
# print(X_src_train[:5])
# print(X_src_train_cu[:5])
# sys.exit()


# adding negatives for RCSLS
Z_src = x_src[:params.maxneg, :]
Z_tgt = x_tgt[:params.maxneg, :]

# initialization:
R = procrustes(X_src_train, Y_tgt_train, gpuid)

R_np = to_numpy(R, gpuid>=0)
nnacc = compute_nn_accuracy(np.dot(x_src_np, R_np.T), x_tgt_np, src2tgt, lexicon_size=lexicon_size)
print("[init -- Procrustes] NN: %.4f"%(nnacc))


start_time = time.time()


nnacc_csls = compute_csls_accuracy(torch.mm(x_src, R.t()), x_tgt, src2tgt, gpuid=gpuid, lexicon_size=lexicon_size)

print("compute_csls_acc time (minutes): ", (time.time() - start_time) / 60)
print("[init -- Procrustes] CSLS: %.4f"%(nnacc_csls))
sys.stdout.flush()

# optimization
fold, Rold = 0, []
niter, lr = params.niter, params.lr

opt_start_time = time.time()

for it in range(0, niter + 1):
    if lr < 1e-4:
        break

    if params.sgd:
        indices = np.random.choice(X_src_train.shape[0], size=params.batchsize, replace=False)
        f, df = rcsls(X_src_train[indices, :], Y_tgt_train[indices, :], Z_src, Z_tgt, R, params.knn)
    else:
        f, df = rcsls(X_src_train, Y_tgt_train, Z_src, Z_tgt, R, params.knn)

    if params.reg > 0:
        print(type(lr))
        print(type(params.reg))
        R *= (1 - lr * params.reg)
    R -= lr * df
    if params.model == "spectral":
        R = proj_spectral(R)

    print("[it=%d] f = %.4f" % (it, f))
    sys.stdout.flush()

    if f > fold and it > 0 and not params.sgd:
        lr /= 2
        f, R = fold, Rold

    fold, Rold = f, R

    # if (it > 0 and it % 10 == 0) or it == niter:
        # nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size)
        # print("[it=%d] NN = %.4f - Coverage = %.4f" % (it, nnacc, len(src2tgt) / lexicon_size))

opt_end_time = time.time()
print("Optimization total time (minutes): ", (opt_end_time - opt_start_time) / 60)

nnacc = compute_nn_accuracy(to_numpy(torch.mm(x_src, R.t()), gpuid>=0), 
                            to_numpy(x_tgt, gpuid>=0), 
                            src2tgt, lexicon_size=lexicon_size)
print("[final] NN = %.4f - Coverage = %.4f" % (nnacc, len(src2tgt) / lexicon_size))
nnacc_csls = compute_csls_accuracy(torch.mm(x_src, R.t()), x_tgt, src2tgt, gpuid=gpuid, lexicon_size=lexicon_size)
print("[final] CSLS = %.4f - Coverage = %.4f" % (
    nnacc_csls, len(src2tgt) / lexicon_size))

if params.output != "":
    print("Saving all aligned vectors at %s" % params.output)
    print("-computing")
    words_full, x_full = load_vectors(params.src_emb, maxload=-1, center=params.center, verbose=False)
    R_np = to_numpy(R, gpuid>=0)
    x = np.dot(x_full, R_np.T)
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8

    print("-saving")
    save_vectors(params.output, x, words_full)
    save_matrix(params.output + "-mat",  R_np)
