from typing import Union, Tuple, Optional, Any, Callable, List
from typing_extensions import Literal
import eagerpy as ep
import logging

from foolbox.devutils import flatten, atleast_kd
from foolbox.types import Bounds
from foolbox.models import Model
from foolbox.criteria import Criterion
from foolbox.distances import l2, linf
from foolbox.attacks.base import MinimizationAttack, T, get_criterion, get_is_adversarial, raise_if_kwargs
from abc import ABC, abstractmethod

from copy import deepcopy
from scipy.fftpack import idct


class QFoolAttack(MinimizationAttack):
    """A decision-based adversarial attack that uses a small number of queries.

    This is the reference implementation for the attack. [#Liu19]_

    Notes:
        Differences to the original reference implementation:
        * The hyperparameters "tolerance", "epsilon", "init_estimation", "omega0", "delta",
            "sigma0" and "k" are chosen based on several experiments
            since no explanations are given about them in the paper

    Args:
        tolerance       : Tolerance of the bisection method.
        epsilon         : Preset threshold for the reduction rate of the perturbation
                            to stop the inner loop.
                            not to be confused with foolbox's "epsilon" of the perturbations.
        max_queries     : Maximum number of queries allowed to use.
        init_estimation : Number of queries used to estimate the gradient in each try.
        omega0          : "omega" is the norm of the perturbation vectors "eta_i",
                            which are used to estimate the gradient.
                            "omega0" is its initial value,
                            and should be at least 2 times the value of "tolerance".
        delta           : Norm of the small perturbation which is used in targeted attacks.
        subspacedim     : Dimension of the subspace to use to create perturbations.
                            Here, 2D DCT bases are used to create low-frequency perturbations.

    References:
        .. [#Liu19] Yujia Liu (*), Seyed-Mohsen Moosavi-Dezfooli, Pascal Frossard,
           "A geometry-inspired decision-based attack",
           https://arxiv.org/abs/1903.10826
    """

    distance = l2

    def __init__(
        self,
        tolerance       : float = 1e-2,
        epsilon         : float = 0.7,
        max_queries     : int   = 1000,
        init_estimation : int   = 100,
        omega0          : float = 3*1e-2,
        delta           : float = 10,
        subspacedim     : int   = None
    ):
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.max_queries = max_queries
        self.init_estimation = init_estimation
        self.omega0 = omega0
        self.delta = delta
        self.subspacedim = subspacedim

        self.phi0 = -1

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        originals, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        criterion = get_criterion(criterion)
        ndim = originals.ndim

        self.non_targeted(originals, model, criterion)

        
    def non_targeted(self,
        images: ep.Tensor,
        model: Model,
        criterion: Criterion
    ):

        # initializations
        is_adversarial = get_is_adversarial(criterion, model)
        bounds = model.bounds

        N, image_shape, dim, labels = process_image_info(images, model)
        if not self.subspacedim:
            self.subspacedim = image_shape[1]
        n = ListOfLists(N)

        # line 1
        loop_i = 0

        # line 2
        P = ListOfLists(N)
        P0, num_calls1 = self.search_P0(images, is_adversarial)
        P0, r, num_calls2 = bisect(images, P0)
        P.append(P0)

        # line 3 and line 14
        x_adv, xi = ListOfLists(N), ListOfLists(N)
        while (ep.sum(n, axis=0) < self.max_queries).any():

            # equation 4
            omega = ep.ones(N) * self.omega0
            phi = ep.ones(N) * self.phi0

            # line 4
            n.append(ep.zeros(N))

            # line 5
            x_adv.append(P.get(loop_i))
            x_adv_prime = P.get(loop_i)

            # line 6 and line 12
            z_dot_eta = ep.zeros_like(images)
            xi.append(ep.zeros_like(images))
            while self.termination_condition(P.get(loop_i), n.get(loop_i), x_adv_prime, x_adv.get(loop_i)):
                
                # line 7
                z_dot_eta_new, rho, num_calls3 = self.estimate_gradient(P.get(loop_i), omega, phi)
                z_dot_eta += z_dot_eta_new

                # line 8
                x_adv_prime = x_adv.get(loop_i)

                # line 9
                n.set(loop_i, n.get(loop_i) + num_calls3)   # n.set(loop_i, n.get(loop_i) + self.init_estimation)

                # line 10
                xi.set(loop_i, normalize(z_dot_eta))

                # line 11
                x_adv_new, num_calls4 = search_boundary_in_direction(images, xi.get(loop_i), r)
                x_adv_new, r, num_calls5 = bisect(images, x_adv_new)
                x_adv.set(loop_i, x_adv_new)

                # equation 4
                omega, phi = update_omega(omega, phi, rho)

            # line 13
            P.append(x_adv.get(loop_i))
            loop_i += 1

        # line 15
        pert_images = P.get(-1)
        r_tot = pert_images - images
        f_pert = query(pert_images)

        return r_tot, ep.sum(n), labels, f_pert, pert_images  # following deepfool's interface


    def process_image_info(images: ep.Tensor,
        model: Model
    ) -> Tuple[int, Tuple[int, int, int], int, ep.Tensor]:
        N = len(images)
        image_shape = images.numpy().shape
        dim = 1
        for s in image_shape:
            dim *= s
        labels = query(model, images)
        return N, image_shape, dim, labels


    def search_P0(self
    ) -> Tuple[ep.Tensor, List[int]]:
        P0 = deepcopy(images)
        sigma0, k = self.P0_constants()
        sigma = sigma0
        num_calls = 0

        while not is_adversarial(P0) and num_calls < k:
            rj = self.create_perturbation() * sigma
            P0 = add_noise(images, rj)
            num_calls += 1
            sigma = sigma0 * (num_calls+1)

        if num_calls == k:
            print("cannot find P0!")

        return P0, num_calls


    def P0_constants(self):
        sigma0 = 0.02
        k = 100
        return sigma0, k


    def termination_condition(self,
        Pi: ep.Tensor,
        ni: List[int],
        x_adv_prime: ep.Tensor,
        x_adv_i: ep.Tensor,
        n: ListOfLists
    ):
        cond1 = ep.sum(n) < self.max_queries

        # note that "ni == 0" is redundant, since "x_adv_prime == Pi" includes that
        if ep.norms.l2(x_adv_prime-Pi) == 0 or ni == 0:
            return cond1

        newrate = ep.norms.l2(x_adv_i-Pi) / (ni + self.init_estimation)
        prevrate = ep.norms.l2(x_adv_prime-Pi) / ni

        print(f"newrate/prevrate = {newrate/prevrate}")
        cond2 = newrate <= self.epsilon * prevrate
        return cond1 and cond2


    def estimate_gradient(self,
        Pi: ep.Tensor,
        omega: ep.Tensor,
        phi: ep.Tensor
    ) -> Tuple[ep.Tensor, List[float], List[int]]:
        z_dot_eta, cnt, num_calls = 0, 0, 0

        for _ in range(self.init_estimation):
            eta = self.create_perturbation()
            eta = self.normalize(eta) * omega
            P_eta = self.add_noise(Pi, eta)
            if self.is_adversarial(P_eta):
                z = 1
                cnt += 1
            else:
                z = -1
            z_dot_eta += z * eta
            num_calls += 1
        
        rho = 0.5 - cnt / self.init_estimation
        # num_calls = self.init_estimation
        return z_dot_eta, rho, num_calls


    def update_omega(self,
        omega: List[float],
        phi: List[float],
        rho: List[float]
    ) -> Tuple[List[float], List[float]]:
        if rho > 0:
            new_phi = -phi
        else:
            new_phi = phi
        new_omega = omega * (1+phi*rho)
        return new_omega, new_phi


    def bisect(self,
        images: ep.Tensor,
        adv_images: ep.Tensor
    ) -> Tuple[ep.Tensor, List[float], List[int]]:
        x = deepcopy(images)
        x_tilde = deepcopy(adv_image)
        num_calls = 0

        while ep.norms.l2(x - x_tilde) > self.tolerance:
            x_mid = (x + x_tilde)/2
            if self.is_adversarial(x_mid):
                x_tilde = x_mid
            else:
                x = x_mid
            num_calls += 1
        return x_tilde, ep.norms.l2(x_tilde - images), num_calls


    def create_perturbation(self
    ) -> ep.Tensor:
        pert_tensor = ep.zeros_like(self.images)

        # added after implementing subspaces
        subspace_shape = (self.image_shape[0], self.subspacedim, self.subspacedim)
        pert_tensor[:, :self.subspacedim, :self.subspacedim] = ep.normal(shape=subspace_shape)

        if self.subspacedim < self.image_shape[1]:
            pert_tensor = ep.from_numpy(idct(idct(pert_tensor.numpy(), axis=2, norm='ortho'),
                axis=1, norm='ortho'))

        return pert_tensor


    def search_boundary_in_direction(self,
        images: ep.Tensor,
        xi: ep.Tensor,
        r: ep.Tensor
    ) -> Tuple[ep.Tensor, List[int]]:
        x_adv = self.add_noise(images, xi*r)
        num_calls = 1
        while not self.is_adversarial(x_adv).all():
            num_calls += 1
            x_adv = self.add_noise(images, xi*num_calls*r)
        return x_adv, num_calls


    def add_noise(self,
        images: ep.Tensor,
        pert: ep.Tensor
    ) -> ep.Tensor:
        min_, max_ = self.bounds
        x_tilde = images + pert
        return ep.clip(x_tilde, min_, max_)


    def normalize(self,
        vec: ep.Tensor
    ) -> ep.Tensor:
        return vec / ep.norms.l2(flatten(vec), axis=-1)


    def query(self,
        images: ep.Tensor
    ) -> ep.Tensor:
        return self.model(images).argmax(axis=-1)


class ListOfLists():
    def __init__(self,
        N: int
    ):
        self.N = N
        self.list = [[] for _ in range(N)]


    def append(self,
        new_list: List
    ):
        for idx in range(self.N):
            self.list[idx].append(new_list[idx])


    def get(self,
        i: int
    ) -> List:
        return [l[i] for l in self.list]


    def set(self,
        i: int,
        x: T
    ):
        for idx in range(self.N):
            self.list[idx][i] = x