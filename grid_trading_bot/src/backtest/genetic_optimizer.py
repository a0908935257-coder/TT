"""
Genetic Algorithm Optimizer.

Implements a genetic algorithm for parameter optimization.
Supports multiple selection, crossover, and mutation methods.
"""

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .optimizer import (
    Optimizer,
    OptimizationDirection,
    OptimizationMethod,
    OptimizationResult,
    OptimizationTrial,
    ParameterSpace,
)


class SelectionMethod(str, Enum):
    """Selection method for choosing parents."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"


class CrossoverMethod(str, Enum):
    """Crossover method for combining parents."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    BLX_ALPHA = "blx_alpha"  # Blend crossover


class MutationMethod(str, Enum):
    """Mutation method for introducing variation."""
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    ADAPTIVE = "adaptive"


@dataclass
class GAConfig:
    """
    Configuration for Genetic Algorithm Optimizer.

    Attributes:
        population_size: Number of individuals in population
        elite_size: Number of top individuals to preserve
        tournament_size: Size for tournament selection
        crossover_rate: Probability of crossover
        mutation_rate: Base probability of mutation
        selection_method: Method for parent selection
        crossover_method: Method for combining parents
        mutation_method: Method for mutation
        adaptive_mutation: Enable adaptive mutation rate
        min_mutation_rate: Minimum mutation rate for adaptive
        max_mutation_rate: Maximum mutation rate for adaptive
        stagnation_generations: Generations without improvement before adapting
        seed: Random seed for reproducibility
    """
    population_size: int = 50
    elite_size: int = 5
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    adaptive_mutation: bool = True
    min_mutation_rate: float = 0.01
    max_mutation_rate: float = 0.5
    stagnation_generations: int = 5
    seed: Optional[int] = None


@dataclass
class Individual:
    """
    Represents an individual in the population.

    Attributes:
        genes: Parameter values
        fitness: Objective function score
        metrics: Additional metrics from evaluation
    """
    genes: dict[str, Any]
    fitness: float = float("-inf")
    metrics: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Individual":
        """Create a copy of this individual."""
        return Individual(
            genes=self.genes.copy(),
            fitness=self.fitness,
            metrics=self.metrics.copy(),
        )


class GeneticAlgorithmOptimizer(Optimizer):
    """
    Genetic Algorithm Optimizer.

    Implements evolutionary search for parameter optimization using:
    - Selection: Tournament, Roulette Wheel, or Rank-based
    - Crossover: Single-point, Two-point, Uniform, or BLX-alpha
    - Mutation: Uniform, Gaussian, or Adaptive

    Example:
        config = GAConfig(
            population_size=50,
            selection_method=SelectionMethod.TOURNAMENT,
            crossover_method=CrossoverMethod.UNIFORM,
            mutation_method=MutationMethod.GAUSSIAN,
        )
        optimizer = GeneticAlgorithmOptimizer(config)
        result = optimizer.optimize(
            objective_fn=my_objective,
            param_space=my_params,
            n_trials=500,
        )
    """

    def __init__(self, config: Optional[GAConfig] = None):
        """
        Initialize genetic algorithm optimizer.

        Args:
            config: GA configuration (uses defaults if None)
        """
        self._config = config or GAConfig()

        if self._config.seed is not None:
            random.seed(self._config.seed)

        # Adaptive mutation state
        self._current_mutation_rate = self._config.mutation_rate
        self._best_fitness_history: list[float] = []
        self._generations_without_improvement = 0

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], tuple[float, dict[str, Any]]],
        param_space: list[ParameterSpace],
        n_trials: int = 100,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        n_jobs: int = 1,
        callback: Optional[Callable[[OptimizationTrial], None]] = None,
    ) -> OptimizationResult:
        """
        Run genetic algorithm optimization.

        Args:
            objective_fn: Function(params) -> (score, metrics)
            param_space: Parameter space definitions
            n_trials: Total number of function evaluations
            direction: Maximize or minimize
            n_jobs: Number of parallel workers
            callback: Optional callback after each trial

        Returns:
            OptimizationResult with best parameters
        """
        start_time = time.time()

        # Initialize population
        population = self._initialize_population(param_space)

        # Evaluate initial population
        trials: list[OptimizationTrial] = []
        trial_number = [0]  # Mutable counter for parallel evaluation

        population = self._evaluate_population(
            population, objective_fn, trials, trial_number, callback, n_jobs
        )

        # Calculate number of generations
        evals_per_generation = self._config.population_size - self._config.elite_size
        remaining_evals = n_trials - self._config.population_size
        n_generations = max(1, remaining_evals // evals_per_generation)

        # Track best
        best_individual = self._get_best(population, direction)

        # Evolution loop
        generation = 0
        while trial_number[0] < n_trials and generation < n_generations:
            generation += 1

            # Create next generation
            new_population = self._evolve(
                population, param_space, direction
            )

            # Evaluate new individuals
            new_population = self._evaluate_population(
                new_population, objective_fn, trials, trial_number, callback, n_jobs
            )

            # Combine with elite
            elite = self._select_elite(population, direction)
            population = elite + new_population

            # Update best
            gen_best = self._get_best(population, direction)
            if self._is_better(gen_best.fitness, best_individual.fitness, direction):
                best_individual = gen_best
                self._generations_without_improvement = 0
            else:
                self._generations_without_improvement += 1

            # Adaptive mutation
            if self._config.adaptive_mutation:
                self._adapt_mutation_rate(direction)

            # Early stopping check
            if trial_number[0] >= n_trials:
                break

        # Create result
        return OptimizationResult(
            best_params=best_individual.genes,
            best_score=best_individual.fitness,
            best_metrics=best_individual.metrics,
            all_trials=trials,
            method=OptimizationMethod.GENETIC,
            direction=direction,
            total_duration_seconds=time.time() - start_time,
        )

    def _initialize_population(self, param_space: list[ParameterSpace]) -> list[Individual]:
        """Initialize random population."""
        population = []
        for _ in range(self._config.population_size):
            genes = {p.name: p.sample_random() for p in param_space}
            population.append(Individual(genes=genes))
        return population

    def _evaluate_population(
        self,
        population: list[Individual],
        objective_fn: Callable[[dict[str, Any]], tuple[float, dict[str, Any]]],
        trials: list[OptimizationTrial],
        trial_number: list[int],
        callback: Optional[Callable[[OptimizationTrial], None]],
        n_jobs: int,
    ) -> list[Individual]:
        """Evaluate fitness of unevaluated individuals."""
        unevaluated = [ind for ind in population if ind.fitness == float("-inf")]

        if not unevaluated:
            return population

        def evaluate_individual(ind: Individual) -> Individual:
            trial_start = time.time()
            score, metrics = objective_fn(ind.genes)
            duration = time.time() - trial_start
            ind.fitness = score
            ind.metrics = metrics
            return ind, duration

        if n_jobs == 1:
            for ind in unevaluated:
                ind, duration = evaluate_individual(ind)

                trial = OptimizationTrial(
                    trial_number=trial_number[0],
                    params=ind.genes.copy(),
                    score=ind.fitness,
                    metrics=ind.metrics.copy(),
                    duration_seconds=duration,
                )
                trials.append(trial)
                trial_number[0] += 1

                if callback:
                    callback(trial)
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(evaluate_individual, ind): ind
                    for ind in unevaluated
                }
                for future in as_completed(futures):
                    ind, duration = future.result()

                    trial = OptimizationTrial(
                        trial_number=trial_number[0],
                        params=ind.genes.copy(),
                        score=ind.fitness,
                        metrics=ind.metrics.copy(),
                        duration_seconds=duration,
                    )
                    trials.append(trial)
                    trial_number[0] += 1

                    if callback:
                        callback(trial)

        return population

    def _evolve(
        self,
        population: list[Individual],
        param_space: list[ParameterSpace],
        direction: OptimizationDirection,
    ) -> list[Individual]:
        """Create new generation through selection, crossover, and mutation."""
        new_size = self._config.population_size - self._config.elite_size
        new_population = []

        while len(new_population) < new_size:
            # Selection
            parent1 = self._select_parent(population, direction)
            parent2 = self._select_parent(population, direction)

            # Crossover
            if random.random() < self._config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, param_space)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            child1 = self._mutate(child1, param_space)
            child2 = self._mutate(child2, param_space)

            # Reset fitness for new individuals
            child1.fitness = float("-inf")
            child2.fitness = float("-inf")

            new_population.append(child1)
            if len(new_population) < new_size:
                new_population.append(child2)

        return new_population

    def _select_parent(
        self,
        population: list[Individual],
        direction: OptimizationDirection,
    ) -> Individual:
        """Select a parent using the configured method."""
        if self._config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population, direction)
        elif self._config.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection(population, direction)
        else:  # RANK
            return self._rank_selection(population, direction)

    def _tournament_selection(
        self,
        population: list[Individual],
        direction: OptimizationDirection,
    ) -> Individual:
        """Select parent via tournament."""
        tournament = random.sample(population, min(self._config.tournament_size, len(population)))
        return self._get_best(tournament, direction)

    def _roulette_selection(
        self,
        population: list[Individual],
        direction: OptimizationDirection,
    ) -> Individual:
        """Select parent via roulette wheel."""
        # Handle negative fitness values
        min_fitness = min(ind.fitness for ind in population)
        offset = abs(min_fitness) + 1 if min_fitness < 0 else 0

        if direction == OptimizationDirection.MAXIMIZE:
            fitness_values = [ind.fitness + offset for ind in population]
        else:
            max_fitness = max(ind.fitness for ind in population)
            fitness_values = [max_fitness - ind.fitness + offset + 1 for ind in population]

        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            return random.choice(population)

        pick = random.uniform(0, total_fitness)
        current = 0
        for ind, fit in zip(population, fitness_values):
            current += fit
            if current >= pick:
                return ind

        return population[-1]

    def _rank_selection(
        self,
        population: list[Individual],
        direction: OptimizationDirection,
    ) -> Individual:
        """Select parent via rank-based selection."""
        reverse = direction == OptimizationDirection.MAXIMIZE
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=reverse)

        # Assign ranks (best = highest rank)
        n = len(sorted_pop)
        ranks = list(range(1, n + 1))
        total_rank = sum(ranks)

        pick = random.uniform(0, total_rank)
        current = 0
        for ind, rank in zip(sorted_pop, ranks):
            current += rank
            if current >= pick:
                return ind

        return sorted_pop[-1]

    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        param_space: list[ParameterSpace],
    ) -> tuple[Individual, Individual]:
        """Perform crossover using the configured method."""
        if self._config.crossover_method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2, param_space)
        elif self._config.crossover_method == CrossoverMethod.TWO_POINT:
            return self._two_point_crossover(parent1, parent2, param_space)
        elif self._config.crossover_method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2, param_space)
        else:  # BLX_ALPHA
            return self._blx_alpha_crossover(parent1, parent2, param_space)

    def _single_point_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        param_space: list[ParameterSpace],
    ) -> tuple[Individual, Individual]:
        """Single-point crossover."""
        param_names = [p.name for p in param_space]
        point = random.randint(1, len(param_names) - 1)

        child1_genes = {}
        child2_genes = {}

        for i, name in enumerate(param_names):
            if i < point:
                child1_genes[name] = parent1.genes[name]
                child2_genes[name] = parent2.genes[name]
            else:
                child1_genes[name] = parent2.genes[name]
                child2_genes[name] = parent1.genes[name]

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def _two_point_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        param_space: list[ParameterSpace],
    ) -> tuple[Individual, Individual]:
        """Two-point crossover."""
        param_names = [p.name for p in param_space]
        n = len(param_names)

        point1 = random.randint(0, n - 1)
        point2 = random.randint(0, n - 1)
        if point1 > point2:
            point1, point2 = point2, point1

        child1_genes = {}
        child2_genes = {}

        for i, name in enumerate(param_names):
            if point1 <= i < point2:
                child1_genes[name] = parent2.genes[name]
                child2_genes[name] = parent1.genes[name]
            else:
                child1_genes[name] = parent1.genes[name]
                child2_genes[name] = parent2.genes[name]

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def _uniform_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        param_space: list[ParameterSpace],
    ) -> tuple[Individual, Individual]:
        """Uniform crossover."""
        child1_genes = {}
        child2_genes = {}

        for p in param_space:
            if random.random() < 0.5:
                child1_genes[p.name] = parent1.genes[p.name]
                child2_genes[p.name] = parent2.genes[p.name]
            else:
                child1_genes[p.name] = parent2.genes[p.name]
                child2_genes[p.name] = parent1.genes[p.name]

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def _blx_alpha_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        param_space: list[ParameterSpace],
        alpha: float = 0.5,
    ) -> tuple[Individual, Individual]:
        """BLX-alpha (Blend) crossover for numeric parameters."""
        child1_genes = {}
        child2_genes = {}

        for p in param_space:
            v1 = parent1.genes[p.name]
            v2 = parent2.genes[p.name]

            if p.param_type in ("int", "float"):
                # Numeric blending
                min_val = min(v1, v2)
                max_val = max(v1, v2)
                spread = (max_val - min_val) * alpha

                low = min_val - spread
                high = max_val + spread

                # Clip to parameter bounds
                if p.low is not None:
                    low = max(low, p.low)
                if p.high is not None:
                    high = min(high, p.high)

                # Ensure high > low for random.uniform (can happen with tight bounds)
                if high <= low:
                    c1 = c2 = low if p.low is not None else (v1 + v2) / 2
                else:
                    c1 = random.uniform(low, high)
                    c2 = random.uniform(low, high)

                if p.param_type == "int":
                    c1 = int(round(c1))
                    c2 = int(round(c2))

                child1_genes[p.name] = c1
                child2_genes[p.name] = c2
            else:
                # Non-numeric: random choice
                child1_genes[p.name] = random.choice([v1, v2])
                child2_genes[p.name] = random.choice([v1, v2])

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def _mutate(self, individual: Individual, param_space: list[ParameterSpace]) -> Individual:
        """Apply mutation to an individual."""
        for p in param_space:
            if random.random() < self._current_mutation_rate:
                if self._config.mutation_method == MutationMethod.UNIFORM:
                    individual.genes[p.name] = p.sample_random()
                elif self._config.mutation_method == MutationMethod.GAUSSIAN:
                    individual.genes[p.name] = self._gaussian_mutate(individual.genes[p.name], p)
                else:  # ADAPTIVE
                    individual.genes[p.name] = self._gaussian_mutate(individual.genes[p.name], p)

        return individual

    def _gaussian_mutate(self, value: Any, param: ParameterSpace) -> Any:
        """Apply Gaussian mutation to a value."""
        if param.param_type == "categorical":
            # For categorical, just sample random
            return param.sample_random()

        if param.param_type == "int":
            # Use integer-appropriate mutation
            if param.low is not None and param.high is not None:
                std = (param.high - param.low) / 6  # ~99% within range
            else:
                std = abs(value) * 0.2 if value != 0 else 1

            new_value = value + random.gauss(0, std)
            new_value = int(round(new_value))

            # Clip to bounds
            if param.low is not None:
                new_value = max(int(param.low), new_value)
            if param.high is not None:
                new_value = min(int(param.high), new_value)

            return new_value

        elif param.param_type == "float":
            if param.low is not None and param.high is not None:
                std = (param.high - param.low) / 6
            else:
                std = abs(value) * 0.2 if value != 0 else 0.1

            new_value = value + random.gauss(0, std)

            # Clip to bounds
            if param.low is not None:
                new_value = max(param.low, new_value)
            if param.high is not None:
                new_value = min(param.high, new_value)

            return new_value

        return value

    def _select_elite(
        self,
        population: list[Individual],
        direction: OptimizationDirection,
    ) -> list[Individual]:
        """Select elite individuals to preserve."""
        reverse = direction == OptimizationDirection.MAXIMIZE
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=reverse)
        return [ind.copy() for ind in sorted_pop[: self._config.elite_size]]

    def _get_best(
        self,
        population: list[Individual],
        direction: OptimizationDirection,
    ) -> Individual:
        """Get the best individual from population."""
        if direction == OptimizationDirection.MAXIMIZE:
            return max(population, key=lambda x: x.fitness)
        else:
            return min(population, key=lambda x: x.fitness)

    def _is_better(
        self,
        fitness1: float,
        fitness2: float,
        direction: OptimizationDirection,
    ) -> bool:
        """Check if fitness1 is better than fitness2."""
        if direction == OptimizationDirection.MAXIMIZE:
            return fitness1 > fitness2
        else:
            return fitness1 < fitness2

    def _adapt_mutation_rate(self, direction: OptimizationDirection) -> None:
        """Adapt mutation rate based on convergence."""
        if self._generations_without_improvement >= self._config.stagnation_generations:
            # Increase mutation to escape local optima
            self._current_mutation_rate = min(
                self._config.max_mutation_rate,
                self._current_mutation_rate * 1.5,
            )
        else:
            # Gradually decrease mutation towards exploration
            self._current_mutation_rate = max(
                self._config.min_mutation_rate,
                self._current_mutation_rate * 0.95,
            )
