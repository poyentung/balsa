"""Module for active learning optimisation process."""

from dataclasses import dataclass, field
from numpy.typing import NDArray

import numpy as np

from balsa.utils import sampling_points, SearchMode
from .obj_func import Ackley, Rastrigin, Rosenbrock, Schwefel, Michalewicz, Griewank
from .search_methods.algorithms import (
    BIPOPCMAES,
    IPOPCMAES,
    DualAnnealing,
    DifferentialEvolution,
    CMAES,
    MCMC,
    Shiwa,
)
from .search_methods.doo import DOO
from .search_methods.soo import SOO
from .search_methods.voo import VOO
from .surrogate import (
    AckleySurrogate,
    RastriginSurrogate,
    RosenbrockSurrogate,
    GriewankSurrogate,
    SchwefelSurrogate,
    MichalewiczSurrogate,
    DefaultSurrogate,
)


@dataclass
class OptimisationRegistry:
    """Registry for optimisation functions and search methods.

    This class serves as a central registry for all available optimisation
    functions, search methods, and surrogate models. It also handles
    registration of special methods that require conditional imports.
    """

    FUNCTIONS = {
        "ackley": Ackley,
        "rastrigin": Rastrigin,
        "rosenbrock": Rosenbrock,
        "schwefel": Schwefel,
        "michalewicz": Michalewicz,
        "griewank": Griewank,
    }

    SEARCH_METHODS = {
        "da": DualAnnealing,
        "diff_evo": DifferentialEvolution,
        "cmaes": CMAES,
        "ipopcmaes": IPOPCMAES,
        "bipopcmaes": BIPOPCMAES,
        "mcmc": MCMC,
        "shiwa": Shiwa,
        "doo": DOO,
        "soo": SOO,
        "voo": VOO,
    }

    SURROGATE_MODELS = {
        "ackley_surrogate": AckleySurrogate,
        "rastrigin_surrogate": RastriginSurrogate,
        "rosenbrock_surrogate": RosenbrockSurrogate,
        "griewank_surrogate": GriewankSurrogate,
        "schwefel_surrogate": SchwefelSurrogate,
        "michalewicz_surrogate": MichalewiczSurrogate,
        "default_surrogate": DefaultSurrogate,
    }

    @classmethod
    def register_special_methods(cls, method_name: str) -> None:
        """Register special optimisation methods that require conditional imports."""
        special_methods = {
            "lamcts": ("balsa.search_methods._lamcts", "LaMCTS"),
            "turbo": ("balsa.search_methods._turbo", "TuRBO"),
            "bo": ("balsa.search_methods._bo", "BO"),
            "saasbo": ("balsa.search_methods._saasbo", "SaasBO"),
        }

        if method_name in special_methods:
            module_path, class_name = special_methods[method_name]
            module = __import__(module_path, fromlist=[class_name])
            cls.SEARCH_METHODS[method_name] = getattr(module, class_name)

    @classmethod
    def register_special_functions(cls, func_name: str) -> None:
        """Register special objective functions that require conditional imports."""
        if func_name == "ptycho":
            from .vlab.ptycho import ElectronPtychography

            cls.FUNCTIONS["ptycho"] = ElectronPtychography
        if func_name == "peptide":
            from .vlab.peptide import CyclicPeptide

            cls.FUNCTIONS["peptide"] = CyclicPeptide


@dataclass
class OptimisationConfig:
    """Configuration for active learning optimisation process."""

    dims: int
    search_method: str
    obj_func_name: str
    num_acquisitions: int
    num_samples_per_acquisition: int
    surrogate: str
    num_init_samples: int = 200
    rollout_round: int = 100
    mode: SearchMode = SearchMode.fast
    search_method_args: dict = field(default_factory=dict)
    func_args: dict = field(default_factory=dict)
    surrogate_args: dict = field(default_factory=dict)
    relative_gaussian_noise_std: float = 0.0
    rng_seed: int = 0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.dims <= 0:
            raise ValueError("Dimensions must be positive")
        if self.num_acquisitions <= 0:
            raise ValueError("Number of samples must be positive")


class ActiveLearningPipeline:
    """Main class for running active learning optimisation loops."""

    def __init__(self, config: OptimisationConfig):
        """Initialise active learning loop with given configuration.

        Args:
            config: Configuration object containing optimisation parameters.
        """
        self.config = config
        self._setup_optimisation()
        self._initialise_samples()

    def _setup_optimisation(self) -> None:
        """Set up optimisation components based on configuration."""
        # Register special methods if needed
        OptimisationRegistry.register_special_methods(self.config.search_method)
        OptimisationRegistry.register_special_functions(self.config.obj_func_name)

        if self.config.obj_func_name not in OptimisationRegistry.FUNCTIONS:
            raise ValueError(f"Unknown function: {self.config.obj_func_name}")

        # Initialise optimisation components
        self.search_method = OptimisationRegistry.SEARCH_METHODS[
            self.config.search_method
        ]
        self.obj_func = OptimisationRegistry.FUNCTIONS[self.config.obj_func_name](
            dims=self.config.dims,
            iters=self.config.num_acquisitions,
            func_args=self.config.func_args,
        )
        self.bounds = [
            (float(self.obj_func.lb[i]), float(self.obj_func.ub[i]))
            for i in range(len(self.obj_func.lb))
        ]
        if self.config.surrogate != None:
            self.surrogate = OptimisationRegistry.SURROGATE_MODELS[
                self.config.surrogate
            ](input_dimension=self.config.dims, **self.config.surrogate_args)
        else:
            self.surrogate = None
        self.rollout_round = self.config.rollout_round

    def _initialise_samples(self) -> None:
        """Initialise samples if required by the search method."""
        if self.config.search_method not in ("lamcts", "turbo", "bo", "saasbo"):
            self.input_X, self.input_scaled_y = sampling_points(
                self.obj_func, self.config.num_init_samples, return_scaled=True
            )

    def run(self) -> None:
        """Execute the optimisation process."""
        if self.config.search_method == "Random":
            self._run_random_search()
        elif self.config.search_method in ("lamcts", "turbo", "bo", "saasbo"):
            self._run_special_optimiser()
        else:
            self._run_standard_optimiser()

    def _run_random_search(self) -> None:
        """Execute random search optimisation."""
        out = sampling_points(
            self.obj_func, self.config.num_acquisitions, return_scaled=False
        )
        for x in out[0]:
            self.obj_func(x)

    def _run_special_optimiser(self) -> None:
        """Execute special optimisation methods (lamcts, turbo, bo, saasbo)."""
        if self.config.search_method == "saasbo":
            optimiser = self.search_method(
                f=self.obj_func,
                dims=self.config.dims,
                model=None,
                name=self.config.obj_func_name,
            )
            print(
                f"This optimisation is based on a {self.config.search_method} optimiser"
            )
            optimiser.run(
                num_acquisitions=self.config.num_acquisitions,
                num_samples_per_acquisition=self.config.num_samples_per_acquisition,
                num_init_samples=self.config.num_init_samples,
                **self.config.search_method_args.get(self.config.search_method, {}),
            )
        else:
            optimiser = self.search_method(
                f=self.obj_func,
                dims=self.config.dims,
                model=None,
                name=self.config.obj_func_name,
            )
            print(
                f"This optimisation is based on a {self.config.search_method} optimiser"
            )
            optimiser.run(
                num_samples=self.config.num_acquisitions
                * self.config.num_samples_per_acquisition,
                num_init_samples=self.config.num_init_samples,
                **self.config.search_method_args.get(self.config.search_method, {}),
            )

    def _run_standard_optimiser(self) -> None:
        """Execute standard optimisation process."""
        for _ in range(self.config.num_acquisitions):
            model = self.surrogate.train_and_evaluate(self.input_X, self.input_scaled_y)
            optimiser = self.search_method(
                f=self.obj_func,
                dims=self.config.dims,
                model=model,
                name=self.config.obj_func_name,
                num_samples_per_acquisition=self.config.num_samples_per_acquisition,
            )
            optimiser.mode = self.config.mode
            print(
                f"This optimisation is based on a {optimiser.mode} mode "
                f"{self.config.search_method} optimiser"
            )

            top_X = optimiser.rollout(
                self.input_X,
                self.input_scaled_y,
                self.rollout_round,
                method_args=self.config.search_method_args.get(
                    self.config.search_method, {}
                ),
            )

            top_scaled_y = self._evaluate_points(top_X)
            self._update_samples(top_X, top_scaled_y)

            if self._break_condition():
                break

    def _evaluate_points(self, points: NDArray) -> NDArray:
        """Evaluate function values for given points."""
        return np.array([self.obj_func(xx, return_scaled=True) for xx in points])

    def _update_samples(self, new_X: NDArray, new_y: NDArray) -> None:
        """Update sample arrays with new points and their values."""
        self.input_X = np.concatenate((self.input_X, new_X), axis=0)
        self.input_scaled_y = np.concatenate((self.input_scaled_y, new_y))

    def _break_condition(self):
        max_scaled_y_idx = np.argmax(self.input_scaled_y)
        unscaled_y = self.obj_func(self.input_X[max_scaled_y_idx], return_scaled=False)
        return np.isclose(unscaled_y, 0.0)
