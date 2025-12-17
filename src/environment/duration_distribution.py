# This file is a modified version of the original file in the AgentSim repo (https://github.com/lukaskirchdorfer/AgentSimulator)
# I've changed the code for better type safety/hinting and improve readability.

import numpy as np
from typing import List, Optional, Union
import math
import sys
from collections import Counter
from enum import Enum

import scipy.stats as st
from scipy.stats import wasserstein_distance


def remove_outliers(data: List[float], m: float = 20.0) -> List[float]:
    """
    Remove outliers from a list of values following the approach presented in https://stackoverflow.com/a/16562028.
    :param data: list of values.
    :param m: maximum ratio between the difference (value - median) and the median of these differences, to NOT be
    considered an outlier. Decreasing the [m] ratio increases the number of detected outliers (observations closer
    to the median are considered as outliers).
    :return: the received list of values without the outliers.
    """
    # Compute distance of each value from the median
    data_array = np.array(data, dtype=np.float64)
    d = np.abs(data_array - np.median(data_array))
    # Compute the median of these distances
    mdev = np.median(d)
    # Compute the ratio between each distance and the median of distances
    s = d / (mdev if mdev else 1.0)
    # Keep values with a ratio lower than the specified threshold
    return data_array[s < m].tolist()


class DistributionType(Enum):
    UNIFORM = "uniform"
    NORMAL = "norm"
    TRIANGULAR = "triang"
    EXPONENTIAL = "expon"
    LOG_NORMAL = "lognorm"
    GAMMA = "gamma"
    FIXED = "fix"

    @staticmethod
    def from_string(value: str) -> "DistributionType":
        name = value.lower()
        if name == "uniform":
            return DistributionType.UNIFORM
        elif name in ("norm", "normal"):
            return DistributionType.NORMAL
        elif name in ("triang", "triangular"):
            return DistributionType.TRIANGULAR
        elif name in ("expon", "exponential"):
            return DistributionType.EXPONENTIAL
        elif name in ("lognorm", "log_normal", "lognormal"):
            return DistributionType.LOG_NORMAL
        elif name == "gamma":
            return DistributionType.GAMMA
        elif name in ["fix", "fixed"]:
            return DistributionType.FIXED
        else:
            raise ValueError(f"Unknown distribution: {value}")


class DurationDistribution:
    def __init__(
        self,
        name: Union[str, DistributionType] = "fix",
        mean: Optional[float] = None,
        var: Optional[float] = None,
        std: Optional[float] = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ):
        self.type = (
            DistributionType.from_string(name) if isinstance(name, str) else name
        )
        self.mean = mean
        self.var = var
        self.std = std
        self.min = minimum
        self.max = maximum

    def generate_sample(self, size: int) -> List[float]:
        """
        Generates a sample of [size] elements following this (self) distribution parameters. The elements are
        positive, and within the limits [self.min, self.max].

        :param size: number of elements to add to the sample.
        :return: list with the elements of the sample.
        """
        # Instantiate empty sample list
        sample: List[float] = []
        i = 0  # flag for iteration limit
        # Generate until full of values within limits
        while len(sample) < size and i < 100:
            # Generate missing elements
            local_sample = self._generate_raw_sample(size - len(sample))
            # Filter out negative and out of limits elements
            local_sample = [element for element in local_sample if element >= 0.0]
            if self.min is not None:
                local_sample = [
                    element for element in local_sample if element >= self.min
                ]
            if self.max is not None:
                local_sample = [
                    element for element in local_sample if element <= self.max
                ]
            # Add generated elements to sample
            sample += local_sample
            i += 1
        # Check if all elements got generated
        if len(sample) < size:
            default_value = self._replace_out_of_bounds_value()
            # print(f"Warning when generating sample of distribution {self}. "
            #       "Too many iterations generating durations out of the distribution limits! "
            #       f"Filling missing values with default ({default_value})!")
            sample += [default_value] * (size - len(sample))
        # Return complete sample
        return sample

    def _generate_raw_sample(self, size: int) -> List[float]:
        """
        Generates a sample of [size] elements following this (self) distribution parameters not ensuring that the
        returned elements are within the interval [self.min, self.max] and positive.

        :param size: number of elements to add to the sample.
        :return: list with the elements of the sample.
        """

        def ensure_list(samples):
            if isinstance(samples, (int, float)):
                return [float(samples)]
            return (
                [float(x) for x in samples.tolist()]
                if hasattr(samples, "tolist")
                else list(samples)
            )

        if self.type == DistributionType.FIXED:
            if self.mean is None:
                raise ValueError("Mean must be provided for fixed distribution")
            return [self.mean] * size
        elif self.type == DistributionType.EXPONENTIAL:
            if self.mean is None or self.min is None:
                raise ValueError(
                    "Mean and min must be provided for exponential distribution"
                )
            scale = self.mean - self.min
            if scale < 0.0:
                print(
                    "Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value."
                )
                scale = self.mean
            samples = st.expon.rvs(loc=self.min, scale=scale, size=size)
            return ensure_list(samples)
        elif self.type == DistributionType.NORMAL:
            if self.mean is None or self.std is None:
                raise ValueError(
                    "Mean and std must be provided for normal distribution"
                )
            samples = st.norm.rvs(loc=self.mean, scale=self.std, size=size)
            return ensure_list(samples)
        elif self.type == DistributionType.UNIFORM:
            if self.min is None or self.max is None:
                raise ValueError(
                    "Min and max must be provided for uniform distribution"
                )
            samples = st.uniform.rvs(loc=self.min, scale=self.max - self.min, size=size)
            return ensure_list(samples)
        elif self.type == DistributionType.LOG_NORMAL:
            if self.mean is None or self.var is None:
                raise ValueError(
                    "Mean and var must be provided for log-normal distribution"
                )

            # Add safety checks for log-normal parameters
            if self.mean <= 0 or self.var <= 0:
                raise ValueError(
                    "Mean and variance must be positive for log-normal distribution"
                )

            pow_mean = float(self.mean) ** 2
            phi = math.sqrt(self.var + pow_mean)

            # Additional safety checks
            if phi <= 0 or pow_mean <= 0:
                raise ValueError("Invalid parameters for log-normal distribution")

            mu = math.log(pow_mean / phi)
            sigma = math.sqrt(math.log(phi**2 / pow_mean))

            # Ensure sigma is positive and reasonable
            if sigma <= 0 or not math.isfinite(sigma):
                raise ValueError("Invalid sigma parameter for log-normal distribution")

            scale = math.exp(mu)
            if scale <= 0 or not math.isfinite(scale):
                raise ValueError("Invalid scale parameter for log-normal distribution")

            samples = st.lognorm.rvs(sigma, loc=0, scale=scale, size=size)
            return ensure_list(samples)
        elif self.type == DistributionType.GAMMA:
            if self.mean is None or self.var is None:
                raise ValueError("Mean and var must be provided for gamma distribution")

            # Add safety checks for gamma parameters
            if self.mean <= 0 or self.var <= 0:
                raise ValueError(
                    "Mean and variance must be positive for gamma distribution"
                )

            shape = float(self.mean) ** 2 / self.var
            scale = self.var / self.mean

            # Ensure shape and scale are positive
            if shape <= 0 or scale <= 0:
                raise ValueError("Invalid parameters for gamma distribution")

            samples = st.gamma.rvs(shape, loc=0, scale=scale, size=size)
            return ensure_list(samples)
        else:
            raise ValueError(f"Unsupported distribution type: {self.type}")

    def _replace_out_of_bounds_value(self) -> float:
        new_value: Optional[float] = None
        if self.mean is not None and self.mean >= 0.0:
            # Set to mean
            new_value = self.mean
        if self.min is not None and self.max is not None:
            # If we have boundaries, check that mean is inside
            if new_value is None or new_value < self.min or new_value > self.max:
                # Invalid, set to middle between min and max
                new_value = self.min + ((self.max - self.min) / 2)
        if new_value is None or new_value < 0.0:
            new_value = 0.0
        # Return fixed value
        return new_value

    def scale_distribution(self, alpha: float) -> "DurationDistribution":
        return DurationDistribution(
            name=self.type,
            mean=(
                self.mean * alpha if self.mean is not None else None
            ),  # Mean: scaled by multiplying by [alpha]
            var=(
                self.var * alpha * alpha if self.var is not None else None
            ),  # Variance: scaled by multiplying by [alpha]^2
            std=(
                self.std * alpha if self.std is not None else None
            ),  # STD: scaled by multiplying by [alpha]
            minimum=(
                self.min * alpha if self.min is not None else None
            ),  # Min: scaled by multiplying by [alpha]
            maximum=(
                self.max * alpha if self.max is not None else None
            ),  # Max: scaled by multiplying by [alpha]
        )

    @staticmethod
    def from_dict(distribution_dict: dict) -> "DurationDistribution":
        """
        Deserialize function distribution provided as a dictionary
        """
        distr_name = DistributionType(distribution_dict["distribution_name"])
        distr_params = distribution_dict["distribution_params"]

        if distr_name == DistributionType.FIXED:
            # no min and max values
            return DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
            )
        elif distr_name == DistributionType.EXPONENTIAL:
            # TODO: discuss whether we need to differentiate min and scale.
            # right now, min is used for calculating the scale
            return DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
                minimum=float(distr_params[1]["value"]),
                maximum=float(distr_params[2]["value"]),
            )
        elif distr_name == DistributionType.UNIFORM:
            return DurationDistribution(
                name=distr_name,
                minimum=float(distr_params[0]["value"]),
                maximum=float(distr_params[1]["value"]),
            )
        elif distr_name == DistributionType.NORMAL:
            return DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
                std=float(distr_params[1]["value"]),
                minimum=float(distr_params[2]["value"]),
                maximum=float(distr_params[3]["value"]),
            )
        elif distr_name == DistributionType.LOG_NORMAL:
            return DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
                var=float(distr_params[1]["value"]),
                minimum=float(distr_params[2]["value"]),
                maximum=float(distr_params[3]["value"]),
            )
        elif distr_name == DistributionType.GAMMA:
            return DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
                var=float(distr_params[1]["value"]),
                minimum=float(distr_params[2]["value"]),
                maximum=float(distr_params[3]["value"]),
            )
        else:
            raise ValueError(f"Unknown distribution type in from_dict: {distr_name}")

    def __str__(self):
        return "DurationDistribution(name: {}, mean: {}, var: {}, std: {}, min: {}, max: {})".format(
            self.type.value, self.mean, self.var, self.std, self.min, self.max
        )


def get_best_fitting_distribution(
    data: List[float],
    filter_outliers: bool = True,
    outlier_threshold: float = 20.0,
) -> DurationDistribution:
    """
    Discover the distribution (exponential, normal, uniform, log-normal, and gamma) that best fits the values in [data].

    :param data: Values to fit a distribution for.
    :param filter_outliers: If true, remove outliers from the sample.
    :param outlier_threshold: Threshold to consider an observation an outlier. Increasing this outlier increases the
                              flexibility of the detection method, i.e., an observation needs to be further from the
                              mean to be considered as outlier.

    :return: the best fitting distribution.
    """
    # Filter outliers
    filtered_data = (
        remove_outliers(data, outlier_threshold) if filter_outliers else data
    )

    # Check for fixed value
    if all(d == 0.0 for d in filtered_data):
        # If it is a fixed value, infer distribution
        return DurationDistribution("fix", 0.0, 0.0, 0.0, 0.0, 0.0)

    # Check if all values are the same (or very close)
    if len(set(filtered_data)) == 1 or (max(filtered_data) - min(filtered_data)) < 1e-6:
        fixed_value = float(filtered_data[0])
        return DurationDistribution(
            "fix", fixed_value, 0.0, 0.0, fixed_value, fixed_value
        )

    # Compute basic statistics
    mean = float(np.mean(filtered_data))
    var = float(np.var(filtered_data))
    std = float(np.std(filtered_data))
    d_min = float(min(filtered_data))
    d_max = float(max(filtered_data))

    # Check for very low variance cases
    if std == 0 or var == 0:
        return DurationDistribution("fix", mean, 0.0, 0.0, d_min, d_max)

    # Create distribution candidates
    dist_candidates = [
        DurationDistribution("expon", mean, var, std, d_min, d_max),
        DurationDistribution("norm", mean, var, std, d_min, d_max),
        DurationDistribution("uniform", mean, var, std, d_min, d_max),
    ]

    # Only add log-normal and gamma if conditions are favorable
    if mean > 0 and var > 0:
        # Additional checks for log-normal distribution
        coefficient_of_variation = std / mean
        if coefficient_of_variation > 0.01:  # Avoid very low variance cases
            try:
                # Test if log-normal parameters would be valid
                pow_mean = float(mean) ** 2
                phi = math.sqrt(var + pow_mean)
                if phi > 0 and pow_mean > 0:
                    mu = math.log(pow_mean / phi)
                    sigma_test = math.sqrt(math.log(phi**2 / pow_mean))
                    if sigma_test > 0 and math.exp(mu) > 0:
                        dist_candidates.append(
                            DurationDistribution(
                                "lognorm", mean, var, std, d_min, d_max
                            )
                        )
            except (ValueError, ZeroDivisionError, OverflowError):
                pass  # Skip log-normal if parameters are invalid

        # Additional checks for gamma distribution
        if var > 0 and mean > 0 and (mean**2 / var) > 0:
            try:
                shape = float(mean) ** 2 / var
                scale = var / mean
                if shape > 0 and scale > 0:
                    dist_candidates.append(
                        DurationDistribution("gamma", mean, var, std, d_min, d_max)
                    )
            except (ValueError, ZeroDivisionError, OverflowError):
                pass  # Skip gamma if parameters are invalid

    # Search for the best one within the candidates
    if not dist_candidates:
        # Fallback to normal distribution
        return DurationDistribution("norm", mean, var, std, d_min, d_max)

    best_distribution = dist_candidates[0]  # Initialize with first candidate
    best_emd = sys.float_info.max

    for distribution_candidate in dist_candidates:
        try:
            # Generate a list of observations from the distribution
            generated_data = distribution_candidate.generate_sample(len(filtered_data))
            # Compute its distance with the observed data
            emd = wasserstein_distance(filtered_data, generated_data)
            # Update the best distribution if better
            if emd < best_emd:
                best_emd = emd
                best_distribution = distribution_candidate
        except Exception:
            # Skip this candidate if it fails to generate samples
            continue

    # Return best distribution
    return best_distribution


def _check_fix(data: List[float], delta: float = 20.0) -> Optional[float]:
    value: Optional[float] = None
    counter = Counter(data)
    for d1 in counter:
        if value is None or (
            counter[d1] > counter[value]
            and sum(1 for d2 in data if abs(d1 - d2) < delta) / len(data) > 0.95
        ):
            # If the value [d1] is more frequent than the current fixed one [value]
            # and
            # the ratio of values similar (or with a difference lower than [delta]) to [d1] is more than 90%
            # update value
            value = d1
    # Return fixed value with more apparitions
    print(f"Fixed value: {value}")
    return value
