"""
Physics-informed degradation models for industrial equipment.

This module provides mathematical degradation models, bearing fault frequency
calculations from physical geometry, and fault signature injection functions.
All models produce synthetic sensor data that is physically plausible for
centrifugal pumps, steam turbines, and electric motors.

References:
    - ISO 13373-1: Condition monitoring and diagnostics of machines
    - ISO 10816: Mechanical vibration evaluation
    - Randall, R.B. "Vibration-based Condition Monitoring" (2011)
"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Degradation models
# ---------------------------------------------------------------------------


def exponential_degradation(
    t: npt.NDArray[np.float64],
    rate: float,
    health_0: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Standard exponential wear model.

    Health decays as H(t) = H0 * exp(-rate * t), clamped to [0, 1].
    This models the common "bathtub curve" wear-out region where degradation
    accelerates as equipment ages. Common in bearings and seals.

    Args:
        t: Time array in hours (1-D or scalar broadcast).
        rate: Degradation rate per hour. Typical values: 1e-5 to 1e-3 depending
            on equipment severity. A rate of 1e-4 means the health drops to
            ~0.37 after 10,000 hours.
        health_0: Initial health value at t=0, default 1.0 (as-new).

    Returns:
        Health values array with same shape as t, clamped to [0.0, health_0].

    Example:
        >>> t = np.linspace(0, 1000, 100)
        >>> health = exponential_degradation(t, rate=0.002)
        >>> health[0]   # 1.0
        >>> health[-1]  # ~0.135
    """
    if rate < 0:
        msg = f"Degradation rate must be non-negative, got {rate}"
        raise ValueError(msg)
    health = health_0 * np.exp(-rate * t)
    return np.clip(health, 0.0, health_0)


def piecewise_linear_degradation(
    t: npt.NDArray[np.float64],
    breakpoints: list[float],
    slopes: list[float],
) -> npt.NDArray[np.float64]:
    """Degradation model with regime changes at specified breakpoints.

    Models equipment that experiences different degradation rates in different
    operating regimes. For example, a pump might degrade slowly during normal
    operation but rapidly after a cavitation event begins.

    The health starts at 1.0 at t=0 and decreases linearly within each segment
    using the corresponding slope. Health is clamped to [0.0, 1.0].

    Args:
        t: Time array in hours (1-D).
        breakpoints: Time values (hours) where the degradation slope changes.
            Must be strictly increasing. The first segment covers [0, breakpoints[0]],
            the second covers [breakpoints[0], breakpoints[1]], etc.
        slopes: Degradation slope (health loss per hour) for each segment.
            Must have len(breakpoints) + 1 elements. Negative slopes indicate
            health decline; positive slopes are clamped to produce health <= 1.0.

    Returns:
        Health values array with same shape as t, clamped to [0.0, 1.0].

    Raises:
        ValueError: If breakpoints and slopes lengths are inconsistent or
            breakpoints are not strictly increasing.

    Example:
        >>> t = np.linspace(0, 1000, 100)
        >>> bp = [400.0]                     # regime change at 400 hours
        >>> slopes = [-0.0001, -0.002]       # slow then fast degradation
        >>> health = piecewise_linear_degradation(t, bp, slopes)
    """
    if len(slopes) != len(breakpoints) + 1:
        msg = (
            f"Number of slopes ({len(slopes)}) must equal "
            f"number of breakpoints + 1 ({len(breakpoints) + 1})"
        )
        raise ValueError(msg)

    # Validate strictly increasing breakpoints
    for i in range(1, len(breakpoints)):
        if breakpoints[i] <= breakpoints[i - 1]:
            msg = f"Breakpoints must be strictly increasing, got {breakpoints}"
            raise ValueError(msg)

    t_arr = np.asarray(t, dtype=np.float64)
    health = np.full_like(t_arr, 1.0, dtype=np.float64)
    segment_starts = [0.0, *breakpoints]
    cumulative_loss = 0.0

    for i, slope in enumerate(slopes):
        seg_start = segment_starts[i]
        seg_end = breakpoints[i] if i < len(breakpoints) else np.inf

        mask = (t_arr > seg_start) & (t_arr <= seg_end)
        if not np.any(mask):
            cumulative_loss += (-slope) * (seg_end - seg_start)
            continue

        health[mask] = 1.0 + cumulative_loss + slope * (t_arr[mask] - seg_start)

        if i < len(breakpoints):
            cumulative_loss += slope * (seg_end - seg_start)

    return np.clip(health, 0.0, 1.0)


def wiener_degradation(
    t: npt.NDArray[np.float64],
    drift: float,
    volatility: float,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Random walk degradation following a Wiener process (Brownian motion).

    Models stochastic degradation where health drifts downward over time with
    random fluctuations. This captures the inherent randomness in wear processes
    where micro-events (debris, thermal cycles) cause unpredictable damage.

    dH = drift * dt + volatility * dW,  starting from H(0) = 1.0.

    Args:
        t: Time array in hours (1-D). Must be non-negative and sorted.
        drift: Expected health decline per hour (must be negative for
            degradation). Typical values: -1e-5 to -1e-3.
        volatility: Standard deviation of random fluctuations per sqrt(hour).
            Typical values: 5e-4 to 5e-2.
        seed: RNG seed for reproducibility.

    Returns:
        Health values array with same shape as t, clamped to [0.0, 1.0].

    Example:
        >>> t = np.linspace(0, 500, 500)
        >>> health = wiener_degradation(t, drift=-0.001, volatility=0.01, seed=42)
    """
    t_arr = np.asarray(t, dtype=np.float64)
    rng = np.random.default_rng(seed)
    dt = np.diff(t_arr, prepend=0.0)
    increments = drift * dt + volatility * np.sqrt(dt) * rng.normal(size=len(t_arr))
    health = 1.0 + np.cumsum(increments)
    return np.clip(health, 0.0, 1.0)


def gamma_degradation(
    t: npt.NDArray[np.float64],
    shape: float,
    scale: float,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Gamma process for modelling cumulative damage.

    The gamma process is widely used in structural reliability for cumulative
    damage because it has independent, non-negative increments -- damage only
    accumulates, never heals. Health starts at 1.0 and damage grows via gamma
    increments: damage ~ Gamma(shape*dt, scale), health = 1.0 - damage_sum.

    Args:
        t: Time array in hours (1-D). Must be non-negative.
        shape: Shape parameter per unit time. Controls rate of damage accumulation.
            The mean damage per hour is shape * scale.
        scale: Scale parameter. Controls variance of damage increments.
        seed: RNG seed for reproducibility.

    Returns:
        Health values array with same shape as t, clamped to [0.0, 1.0].

    Example:
        >>> t = np.linspace(0, 1000, 500)
        >>> health = gamma_degradation(t, shape=0.05, scale=0.0001, seed=42)
        >>> # mean damage per hour = 5e-6, so health~=0.95 after 10000 hours
    """
    t_arr = np.asarray(t, dtype=np.float64)
    rng = np.random.default_rng(seed)
    dt = np.diff(t_arr, prepend=0.0)
    damage = np.zeros(len(t_arr), dtype=np.float64)
    for i in range(1, len(t_arr)):
        damage[i] = damage[i - 1] + rng.gamma(shape=shape * dt[i], scale=scale)
    health = 1.0 - damage
    return np.clip(health, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Bearing fault frequencies
# ---------------------------------------------------------------------------


def bearing_fault_frequencies(
    rpm: float,
    n_balls: int,
    ball_diameter: float,
    pitch_diameter: float,
    contact_angle: float,
) -> dict[str, float]:
    r"""Calculate bearing characteristic fault frequencies from physical geometry.

    Computes the four standard bearing fault frequencies used in vibration
    condition monitoring, as defined by ISO 13373-1 and IEEE standards.

    Args:
        rpm: Shaft rotational speed in revolutions per minute.
        n_balls: Number of rolling elements (balls or rollers).
        ball_diameter: Diameter of a single rolling element, in mm.
        pitch_diameter: Diameter of the pitch circle (center of ball orbit), in mm.
        contact_angle: Contact angle in radians (0 = radial, π/2 = thrust).

    Returns:
        Dictionary with keys (frequency in Hz):
        - "bpfo": Ball Pass Frequency Outer race
        - "bpfi": Ball Pass Frequency Inner race
        - "bsf": Ball Spin Frequency
        - "ftf": Fundamental Train Frequency (cage speed)

    Notes:
        BPFO = (n/2) * fr * (1 - (d/D) * cos(θ))
        BPFI = (n/2) * fr * (1 + (d/D) * cos(θ))
        BSF  = (D/(2d)) * fr * (1 - (d/D)^2 * cos^2(θ))
        FTF  = (fr/2) * (1 - (d/D) * cos(θ))
        where fr = rpm/60 (shaft frequency in Hz), d = ball_diameter,
        D = pitch_diameter, θ = contact_angle.

    Example:
        >>> # 6205 bearing: 9 balls, d=7.94mm, D=39.04mm, radial
        >>> freqs = bearing_fault_frequencies(1772, 9, 7.94, 39.04, 0.0)
        >>> freqs["bpfo"]  # ~65 Hz
    """
    fr = rpm / 60.0  # shaft rotational frequency in Hz
    d = ball_diameter
    D = pitch_diameter
    theta = contact_angle
    cos_theta = np.cos(theta)

    ratio = d / D
    bpfo = (n_balls / 2.0) * fr * (1.0 - ratio * cos_theta)
    bpfi = (n_balls / 2.0) * fr * (1.0 + ratio * cos_theta)
    bsf = (D / (2.0 * d)) * fr * (1.0 - (ratio**2) * (cos_theta**2))
    ftf = (fr / 2.0) * (1.0 - ratio * cos_theta)

    return {
        "bpfo": float(bpfo),
        "bpfi": float(bpfi),
        "bsf": float(bsf),
        "ftf": float(ftf),
    }


# ---------------------------------------------------------------------------
# Fault signature injection
# ---------------------------------------------------------------------------


def add_fault_signature(
    values: npt.NDArray[np.float64],
    fault_mode: str,
    params: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Add characteristic fault patterns to a clean sensor signal.

    Each fault mode produces distinct, physically-motivated signatures
    that a downstream condition monitoring system should detect.

    Args:
        values: Clean sensor signal array (1-D), e.g., vibration in mm/s.
        fault_mode: One of "bearing_wear", "cavitation", "misalignment",
            "seal_leakage", "imbalance". Case-insensitive.
        params: Dictionary of parameters specific to the fault mode:
            - **bearing_wear**: ``rpm`` (float), ``bearing_freqs`` (dict as
              returned by bearing_fault_frequencies), ``amplitude`` (float),
              ``dt`` (float, sample interval in seconds).
            - **cavitation**: ``frequency`` (float, bubble collapse frequency
              in Hz), ``amplitude`` (float), ``dt`` (float).
            - **misalignment**: ``rpm`` (float), ``amplitude`` (float),
              ``dt`` (float).
            - **seal_leakage**: ``offset`` (float, pressure loss in bar),
              ``ramp_rate`` (float, progressive leak rate in bar/hour).
            - **imbalance**: ``rpm`` (float), ``amplitude`` (float),
              ``dt`` (float).

    Returns:
        Signal array (same shape as values) with fault signature added.

    Example:
        >>> t = np.linspace(0, 1, 1000, endpoint=False)
        >>> clean = 2.0 * np.sin(2 * np.pi * 29.5 * t)  # 29.5 Hz shaft
        >>> faulty = add_fault_signature(clean, "misalignment",
        ...     {"rpm": 1770, "amplitude": 1.5, "dt": 0.001})
    """
    mode = fault_mode.lower()
    result = values.copy()

    if mode == "bearing_wear":
        result = _add_bearing_wear(result, params)
    elif mode == "cavitation":
        result = _add_cavitation(result, params)
    elif mode == "misalignment":
        result = _add_misalignment(result, params)
    elif mode == "seal_leakage":
        result = _add_seal_leakage(result, params)
    elif mode == "imbalance":
        result = _add_imbalance(result, params)
    else:
        logger.warning("Unknown fault mode '%s', returning clean signal.", fault_mode)

    return result


def _add_bearing_wear(
    signal: npt.NDArray[np.float64],
    params: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Add bearing wear signature: high-frequency impulses at BPFO/BPFI/BSF.

    Bearing wear produces sharp impact events at the characteristic frequencies.
    We model this as amplitude-modulated pulses at the ball-pass frequencies
    plus raised noise floor (broadband energy from friction).
    """
    n = len(signal)
    dt = float(params["dt"])
    amplitude = float(params.get("amplitude", 0.5))
    freqs = params.get("bearing_freqs", {})

    t = np.arange(n) * dt
    fault_signal = np.zeros(n, dtype=np.float64)

    # Impulse train at BPFO (outer race defect -- most common)
    if "bpfo" in freqs:
        bpfo = float(freqs["bpfo"])
        fault_signal += amplitude * np.abs(np.sin(2.0 * np.pi * bpfo * t))

    # Additional modulation at BPFI
    if "bpfi" in freqs:
        bpfi = float(freqs["bpfi"])
        fault_signal += 0.7 * amplitude * np.abs(np.sin(2.0 * np.pi * bpfi * t))

    # High-frequency carrier from ball spin
    if "bsf" in freqs:
        bsf = float(freqs["bsf"])
        carrier = np.sin(2.0 * np.pi * bsf * t)
        # Amplitude modulation by cage frequency
        if "ftf" in freqs:
            ftf = float(freqs["ftf"])
            envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * ftf * t))
            fault_signal += 0.3 * amplitude * envelope * carrier
        else:
            fault_signal += 0.3 * amplitude * carrier

    return signal + fault_signal


def _add_cavitation(
    signal: npt.NDArray[np.float64],
    params: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Add cavitation signature: broadband random pressure fluctuations.

    Cavitation produces vapor bubble collapse events that create random
    high-frequency pressure transients. We model this as random impulses
    with a Poisson arrival process, superposed on the base signal.
    """
    n = len(signal)
    dt = float(params["dt"])
    amplitude = float(params.get("amplitude", 0.8))
    freq = float(params.get("frequency", 200.0))  # bubble collapse rate ~Hz

    rng = np.random.default_rng(params.get("seed", 42))
    t = np.arange(n) * dt

    # Poisson impulse train
    prob_per_sample = freq * dt
    impulses = rng.random(n) < prob_per_sample
    impulse_amplitudes = rng.exponential(scale=amplitude, size=n)
    cav_signal = impulses.astype(np.float64) * impulse_amplitudes

    # Convolve with a short exponential decay to simulate bubble collapse ringing
    decay_samples = int(0.005 / dt)  # 5 ms decay time
    if decay_samples > 1:
        kernel = np.exp(-np.arange(decay_samples) / (decay_samples / 3))
        kernel /= kernel.sum()
        cav_signal = np.convolve(cav_signal, kernel, mode="same")

    return signal + cav_signal


def _add_misalignment(
    signal: npt.NDArray[np.float64],
    params: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Add misalignment signature: elevated vibration at 2x and 3x RPM.

    Angular or parallel misalignment produces vibration predominantly at
    2x shaft speed (and sometimes 3x for severe misalignment). There is
    also elevated axial vibration (not modeled here in radial signal).
    """
    n = len(signal)
    dt = float(params["dt"])
    rpm = float(params["rpm"])
    amplitude = float(params.get("amplitude", 1.0))

    fr = rpm / 60.0  # shaft frequency in Hz
    t = np.arange(n) * dt

    # 2x component (dominant for angular misalignment)
    fault_signal = amplitude * np.sin(2.0 * np.pi * (2.0 * fr) * t)
    # 3x component (severe misalignment)
    fault_signal += 0.5 * amplitude * np.sin(2.0 * np.pi * (3.0 * fr) * t)
    # 1x component also slightly elevated
    fault_signal += 0.3 * amplitude * np.sin(2.0 * np.pi * fr * t)

    return signal + fault_signal


def _add_seal_leakage(
    signal: npt.NDArray[np.float64],
    params: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Add seal leakage signature: progressive pressure drop.

    Seal leakage typically manifests as a steady or accelerating drop in
    system pressure. We model this as a negative offset that grows over time.
    This is primarily for pressure sensor signals.
    """
    n = len(signal)
    dt = float(params.get("dt", 1.0))
    offset = float(params.get("offset", 0.2))  # initial pressure drop in bar
    ramp_rate = float(params.get("ramp_rate", 0.05))  # bar per hour

    t_hours = np.arange(n) * dt / 3600.0
    pressure_loss = offset + ramp_rate * t_hours

    return signal - pressure_loss


def _add_imbalance(
    signal: npt.NDArray[np.float64],
    params: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Add rotor imbalance signature: elevated vibration at 1x RPM.

    Mass imbalance produces a sinusoidal vibration at the shaft rotational
    frequency (1x). The amplitude is proportional to the square of speed.
    This is the most common vibration fault.
    """
    n = len(signal)
    dt = float(params["dt"])
    rpm = float(params["rpm"])
    amplitude = float(params.get("amplitude", 1.5))

    fr = rpm / 60.0  # shaft frequency in Hz
    t = np.arange(n) * dt

    fault_signal = amplitude * np.sin(2.0 * np.pi * fr * t)
    # Slight harmonic at 2x is also common with imbalance
    fault_signal += 0.1 * amplitude * np.sin(2.0 * np.pi * (2.0 * fr) * t)

    return signal + fault_signal
