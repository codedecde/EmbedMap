"""
A module to combine pytorch functionality to
the pymanopt class, so that I can use pytorch
end to end
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils import to_numpy
from pymanopt.manifolds import Product, Stiefel, PositiveDefinite


class OrthoManifold(object):
    """
    A pytorch wrapper for orthogonal manifold learning
    """

    def __init__(self, Xs, Xt, device=-1):
        self.Xs = Xs
        self.Xt = Xt
        assert isinstance(self.Xs, torch.Tensor)
        assert isinstance(self.Xt, torch.Tensor)
        d1 = self.Xs.size(1)
        d2 = self.Xt.size(1)
        self.device = device
        assert d1 == d2, f"Error. Found different dims {d1}, {d2}"
        self.manifold = Product([Stiefel(d1, d2)])

    def cost(self, param_list):
        W_numpy = param_list[0]
        W = Variable(torch.FloatTensor(W_numpy))
        if self.device >= 0:
            W = W.cuda(self.device)
        cost = ((torch.mm(self.Xs, W) - self.Xt) ** 2).sum()
        return to_numpy(cost, self.device >= 0)

    def egrad(self, param_list):
        W_numpy = param_list[0]
        W = Variable(torch.FloatTensor(W_numpy))
        if self.device >= 0:
            W = W.cuda(self.device)
        W = nn.Parameter(W)
        cost = self.cost_torch(W)
        cost.backward()
        return [to_numpy(W.grad, self.device >= 0)]

    def cost_torch(self, W):
        cost = ((torch.mm(Variable(self.Xs), W) - Variable(self.Xt)) ** 2).sum()
        return cost


class GeomManifold(object):
    """
    A pytorch wrapper for Geometric Manifold learning
    """

    def __init__(self, Xs, Xt, A, lbda, rank, device=-1):
        self.Xs = Xs
        self.Xt = Xt
        self.A = A
        self.rank = rank
        self.lbda = lbda
        assert isinstance(self.Xs, torch.Tensor)
        assert isinstance(self.Xt, torch.Tensor)
        assert isinstance(self.A, torch.Tensor)
        self.device = device

        d1 = self.Xs.size(1)
        d2 = self.Xt.size(1)

        assert (d1 == rank == d2), f"Found dimensions {d1}, {rank}, {d2}"
        d = d1
        self.manifold = Product(
            [Stiefel(d, d), PositiveDefinite(d), Stiefel(d, d)])

    def cost(self, param_list):
        U1, B, U2 = [Variable(torch.FloatTensor(x)) for x in param_list]
        if self.device >= 0:
            U1 = U1.cuda(self.device)
            B = B.cuda(self.device)
            U2 = U2.cuda(self.device)
        cost = self.cost_torch(U1, B, U2)
        return to_numpy(cost, self.device >= 0)

    def egrad(self, param_list):
        U1, B, U2 = [Variable(torch.FloatTensor(x)) for x in param_list]
        if self.device >= 0:
            U1 = U1.cuda(self.device)
            B = B.cuda(self.device)
            U2 = U2.cuda(self.device)
        U1, B, U2 = [nn.Parameter(x) for x in [U1, B, U2]]
        cost = self.cost_torch(U1, B, U2)
        cost.backward()
        return [to_numpy(x.grad, self.device >= 0) for x in [U1, B, U2]]

    def cost_torch(self, U1, B, U2):
        cost = torch.mm(torch.mm(self.Xs, torch.mm(U1, torch.mm(B, U2.t()))),
            self.Xt.t()) - self.A
        cost = (cost ** 2).sum()
        cost += 0.5 * self.lbda * (B ** 2).sum()
        return cost

    def transform_embeddings(self, src_emb, tgt_emb, Us, B, Ut):
        u, s, vh = np.linalg.svd(B, full_matrices=True)
        b_sqrt = np.dot(u, np.dot(np.diag(np.sqrt(s)), vh))
        src_transform = np.dot(np.dot(src_emb, Us), b_sqrt)
        tgt_transform = np.dot(np.dot(tgt_emb, Ut), b_sqrt)
        return src_transform, tgt_transform


class Problem(object):
    """
    A wrapper around the pymanopt class, because its
    automatic imports are annoying, and instantiate
    tensorflow / pytorch even when I don't want them to
    """

    def __init__(self, manifold, cost, egrad=None, ehess=None, grad=None,
                 hess=None, arg=None, precon=None, verbosity=2):
        self.manifold = manifold
        # We keep a reference to the original cost function in case we want to
        # call the `prepare` method twice (for instance, after switching from
        # a first- to second-order method).
        self._cost = None
        self._original_cost = cost
        self._egrad = egrad
        self._ehess = ehess
        self._grad = grad
        self._hess = hess
        self._arg = arg
        self._backend = None
        self.verbosity = verbosity

        if precon is None:
            def precon(x, d):
                return d
        self.precon = precon

    @property
    def backend(self):
        return self._backend

    @property
    def cost(self):
        if self._cost is None and callable(self._original_cost):
            self._cost = self._original_cost
        return self._cost

    @property
    def egrad(self):
        if self._egrad is None:
            raise RuntimeError("You need to specify egrad for Pytorch layers")
        return self._egrad

    @property
    def grad(self):
        if self._grad is None:
            # Explicit access forces computation/compilation if necessary.
            egrad = self._egrad

            def grad(x):
                return self.manifold.egrad2rgrad(x, egrad(x))
            self._grad = grad
        return self._grad

    @property
    def ehess(self):
        raise RuntimeError(
            "Hessian Computation not supported for pytorch layers")

    @property
    def hess(self):
        raise RuntimeError(
            "Hessian Computation not supported for pytorch layers")


if __name__ == "__main__":
    pass
    # manifold_learner = GeomManifold(Xs, Xt, A, lbda, Xs.size(1))
    # # manifold_learner = OrthoManifold(Xs, Xt)
    # problem = Problem(
    #     manifold=manifold_learner.manifold,
    #     cost=manifold_learner.cost,
    #     egrad=manifold_learner.egrad)
    # max_opt_time = 5000
    # max_opt_iter = 150
    # solver = ConjugateGradient(
    #     maxtime=max_opt_time, maxiter=max_opt_iter)
    # theta = solver.solve(problem)
    # # 