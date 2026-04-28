#!/usr/bin/env python3
"""Repo-local mixed-family DAGMA implementation for nearby-peak causal discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from scipy.special import expit, gammaln
from tqdm.auto import tqdm


VALID_FAMILIES = {"gaussian", "bernoulli", "nb2"}


@dataclass
class ColumnFamilyStats:
    family: str
    intercept: float
    dispersion: float | None


class MixedFamilyDagmaLinear:
    """Mixed-family DAGMA with nodewise Gaussian, Bernoulli, and NB2 score terms."""

    def __init__(
        self,
        families: Sequence[str],
        *,
        verbose: bool = False,
        dtype: type = np.float64,
        max_intercept_iter: int = 100,
        intercept_tol: float = 1e-10,
        nb_dispersion_floor: float = 1e-6,
        nb_dispersion_cap: float = 1e6,
    ) -> None:
        self.families = [str(f).lower() for f in families]
        invalid = sorted(set(self.families) - VALID_FAMILIES)
        if invalid:
            raise ValueError(f"Unsupported families: {invalid}")
        self.verbose = verbose
        self.dtype = dtype
        self.max_intercept_iter = int(max_intercept_iter)
        self.intercept_tol = float(intercept_tol)
        self.nb_dispersion_floor = float(nb_dispersion_floor)
        self.nb_dispersion_cap = float(nb_dispersion_cap)
        self.vprint = print if verbose else (lambda *args, **kwargs: None)

    def _h(self, W: np.ndarray, s: float = 1.0) -> tuple[float, np.ndarray]:
        M = s * self.Id - W * W
        h = -la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return float(h), G_h

    def _func(self, W: np.ndarray, mu: float, s: float = 1.0) -> tuple[float, float, float]:
        score, _ = self._score(W)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h
        return float(obj), float(score), float(h)

    def _adam_update(self, grad: np.ndarray, iteration: int, beta_1: float, beta_2: float) -> np.ndarray:
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iteration)
        v_hat = self.opt_v / (1 - beta_2 ** iteration)
        return m_hat / (np.sqrt(v_hat) + 1e-8)

    def _estimate_nb2_dispersion(self, y: np.ndarray) -> float:
        mean = float(np.mean(y))
        var = float(np.var(y, ddof=1)) if y.shape[0] > 1 else mean
        if mean <= 0:
            return self.nb_dispersion_cap
        if var <= mean + self.nb_dispersion_floor:
            return self.nb_dispersion_cap
        theta = mean * mean / max(var - mean, self.nb_dispersion_floor)
        return float(np.clip(theta, self.nb_dispersion_floor, self.nb_dispersion_cap))

    def _solve_gaussian_intercept(self, y: np.ndarray, z: np.ndarray) -> float:
        return float(np.mean(y - z))

    def _solve_bernoulli_intercept(self, y: np.ndarray, z: np.ndarray) -> float:
        intercept = 0.0
        for _ in range(self.max_intercept_iter):
            eta = np.clip(z + intercept, -30.0, 30.0)
            prob = expit(eta)
            grad = float(np.sum(prob - y))
            if abs(grad) <= self.intercept_tol:
                break
            hess = float(np.sum(prob * (1.0 - prob)))
            if hess <= self.intercept_tol:
                break
            step = grad / hess
            intercept -= step
            if abs(step) <= self.intercept_tol:
                break
        return float(intercept)

    def _solve_nb2_intercept(self, y: np.ndarray, z: np.ndarray, theta: float) -> float:
        mean_y = float(np.mean(y))
        base = np.mean(np.exp(np.clip(z, -30.0, 30.0)))
        intercept = float(np.log(mean_y + 1e-6) - np.log(base + 1e-6))
        for _ in range(self.max_intercept_iter):
            eta = np.clip(z + intercept, -30.0, 30.0)
            mu = np.exp(eta)
            grad_vec = (mu - y) / (1.0 + (mu / theta))
            grad = float(np.sum(grad_vec))
            if abs(grad) <= self.intercept_tol:
                break
            hess_vec = theta * mu * (theta + y) / np.square(theta + mu)
            hess = float(np.sum(hess_vec))
            if hess <= self.intercept_tol:
                break
            step = grad / hess
            intercept -= step
            if abs(step) <= self.intercept_tol:
                break
        return float(intercept)

    def _column_score(self, child_idx: int, linear_eta: np.ndarray) -> tuple[float, np.ndarray, ColumnFamilyStats]:
        y = self.X[:, child_idx]
        offset = self.offsets[:, child_idx]
        z = linear_eta + offset
        family = self.families[child_idx]

        if family == "gaussian":
            intercept = self._solve_gaussian_intercept(y, z)
            eta = z + intercept
            resid = eta - y
            loss = 0.5 * float(np.mean(resid * resid))
            grad_eta = resid / float(self.n)
            return loss, grad_eta, ColumnFamilyStats(family=family, intercept=intercept, dispersion=None)

        if family == "bernoulli":
            intercept = self._solve_bernoulli_intercept(y, z)
            eta = np.clip(z + intercept, -30.0, 30.0)
            prob = expit(eta)
            loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
            grad_eta = (prob - y) / float(self.n)
            return loss, grad_eta, ColumnFamilyStats(family=family, intercept=intercept, dispersion=None)

        if family == "nb2":
            theta = float(self.nb2_dispersion_[child_idx])
            intercept = self._solve_nb2_intercept(y, z, theta)
            eta = np.clip(z + intercept, -30.0, 30.0)
            mu = np.exp(eta)
            loss_vec = (
                gammaln(y + theta)
                - gammaln(theta)
                - gammaln(y + 1.0)
                + theta * (np.log(theta + mu) - np.log(theta))
                + y * (np.log(theta + mu) - np.log(np.clip(mu, 1e-12, None)))
            )
            grad_eta = ((mu - y) / (1.0 + (mu / theta))) / float(self.n)
            return float(np.mean(loss_vec)), grad_eta, ColumnFamilyStats(family=family, intercept=intercept, dispersion=theta)

        raise ValueError(f"Unsupported family: {family}")

    def _score(self, W: np.ndarray) -> tuple[float, np.ndarray]:
        linear_pred = self.X @ W
        G_loss = np.zeros_like(W)
        total_loss = 0.0
        family_stats: list[ColumnFamilyStats] = []
        for child_idx in range(self.d):
            loss_j, grad_eta_j, stats_j = self._column_score(child_idx, linear_pred[:, child_idx])
            total_loss += loss_j
            G_loss[:, child_idx] = self.X.T @ grad_eta_j
            family_stats.append(stats_j)
        self.column_family_stats_ = family_stats
        return float(total_loss), G_loss

    def minimize(
        self,
        W: np.ndarray,
        mu: float,
        max_iter: int,
        s: float,
        lr: float,
        tol: float = 1e-6,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        pbar: tqdm | None = None,
    ) -> tuple[np.ndarray, bool]:
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        self.vprint(f"\n\nMinimize with -- mu:{mu} -- lr:{lr} -- s:{s} -- l1:{self.lambda1} for {max_iter} max iterations")

        mask_inc = np.zeros((self.d, self.d), dtype=self.dtype)
        if self.inc_c is not None:
            mask_inc[self.inc_r, self.inc_c] = -2 * mu * self.lambda1
        mask_exc = np.ones((self.d, self.d), dtype=self.dtype)
        if self.exc_c is not None:
            mask_exc[self.exc_r, self.exc_c] = 0.0

        grad = np.zeros_like(W)
        for iteration in range(1, max_iter + 1):
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0):
                if iteration == 1 or s <= 0.9:
                    self.vprint(f"W went out of domain for s={s} at iteration {iteration}")
                    return W, False
                W += lr * grad
                lr *= 0.5
                if lr <= 1e-16:
                    return W, True
                W -= lr * grad
                M = sla.inv(s * self.Id - W * W) + 1e-16
                self.vprint(f"Learning rate decreased to lr: {lr}")

            _, G_score = self._score(W)
            Gobj = mu * (G_score + self.lambda1 * np.sign(W)) + 2 * W * M.T + mask_inc * np.sign(W)
            grad = self._adam_update(Gobj, iteration, beta_1, beta_2)
            W -= lr * grad
            W *= mask_exc
            np.fill_diagonal(W, 0.0)

            if iteration % self.checkpoint == 0 or iteration == max_iter:
                obj_new, score, h = self._func(W, mu, s)
                self.vprint(f"\nInner iteration {iteration}")
                self.vprint(f"\th(W_est): {h:.4e}")
                self.vprint(f"\tscore(W_est): {score:.4e}")
                self.vprint(f"\tobj(W_est): {obj_new:.4e}")
                if np.abs((obj_prev - obj_new) / max(abs(obj_prev), 1e-12)) <= tol:
                    if pbar is not None:
                        pbar.update(max_iter - iteration + 1)
                    break
                obj_prev = obj_new
            if pbar is not None:
                pbar.update(1)
        return W, True

    def fit(
        self,
        X: np.ndarray,
        *,
        offsets: np.ndarray | None = None,
        lambda1: float = 0.03,
        w_threshold: float = 0.3,
        T: int = 5,
        mu_init: float = 1.0,
        mu_factor: float = 0.1,
        s: list[float] | float = [1.0, 0.9, 0.8, 0.7, 0.6],
        warm_iter: int = int(3e4),
        max_iter: int = int(6e4),
        lr: float = 0.0003,
        checkpoint: int = 1000,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        exclude_edges: Iterable[tuple[int, int]] | None = None,
        include_edges: Iterable[tuple[int, int]] | None = None,
        show_progress: bool = True,
        nb2_dispersions: Sequence[float] | None = None,
    ) -> np.ndarray:
        self.X = np.asarray(X, dtype=self.dtype)
        self.n, self.d = self.X.shape
        if len(self.families) != self.d:
            raise ValueError(f"Expected {self.d} family labels, got {len(self.families)}")
        self.lambda1 = float(lambda1)
        self.checkpoint = int(checkpoint)
        self.Id = np.eye(self.d, dtype=self.dtype)

        if offsets is None:
            self.offsets = np.zeros_like(self.X, dtype=self.dtype)
        else:
            self.offsets = np.asarray(offsets, dtype=self.dtype)
            if self.offsets.shape != self.X.shape:
                raise ValueError(f"Offsets shape {self.offsets.shape} did not match X shape {self.X.shape}")

        self.exc_r, self.exc_c = None, None
        self.inc_r, self.inc_c = None, None
        if exclude_edges is not None:
            exclude_edges = list(exclude_edges)
            if exclude_edges:
                self.exc_r, self.exc_c = zip(*exclude_edges)
        if include_edges is not None:
            include_edges = list(include_edges)
            if include_edges:
                self.inc_r, self.inc_c = zip(*include_edges)

        if nb2_dispersions is None:
            disp = []
            for j, family in enumerate(self.families):
                if family == "nb2":
                    disp.append(self._estimate_nb2_dispersion(self.X[:, j]))
                else:
                    disp.append(np.nan)
            self.nb2_dispersion_ = np.asarray(disp, dtype=self.dtype)
        else:
            self.nb2_dispersion_ = np.asarray(nb2_dispersions, dtype=self.dtype)
            if self.nb2_dispersion_.shape != (self.d,):
                raise ValueError("nb2_dispersions must have shape (d,)")

        self.W_est = np.zeros((self.d, self.d), dtype=self.dtype)
        mu = float(mu_init)
        if isinstance(s, list):
            if len(s) < T:
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif isinstance(s, (int, float)):
            s = T * [float(s)]
        else:
            raise ValueError("s should be a list, int, or float.")

        total_iters = (T - 1) * int(warm_iter) + int(max_iter)
        with tqdm(total=total_iters, disable=not show_progress) as pbar:
            for i in range(int(T)):
                self.vprint(f"\nIteration -- {i + 1}:")
                lr_adam, success = float(lr), False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, success = self.minimize(
                        self.W_est.copy(),
                        mu,
                        inner_iters,
                        float(s[i]),
                        lr=lr_adam,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        pbar=pbar,
                    )
                    if success is False:
                        self.vprint("Retrying with larger s")
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                mu *= mu_factor

        self.h_final, _ = self._h(self.W_est)
        self.score_final, _ = self._score(self.W_est)
        self.W_est[np.abs(self.W_est) < w_threshold] = 0.0
        return self.W_est
