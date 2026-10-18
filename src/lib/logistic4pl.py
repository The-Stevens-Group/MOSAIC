import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

import pandas as pd
import os


def logistic4pl(x, bottom, top, logEC50, hill_slope):
    """
    Docstring for logistic4pl

    :param x: Description
    :param bottom: Description
    :param top: Description
    :param logEC50: Description
    :param hill_slope: Description
    """

    x = np.asarray(x, dtype=float)
    # If x <= 0, replace with a very small positive number to avoid log10(0) or log10(negative).
    safe_x = np.where(x <= 0, 1e-12, x)
    return bottom + (top - bottom) / (
        1 + 10 ** ((logEC50 - np.log10(safe_x)) * hill_slope)
    )


def fit_logistic4pl(true_conc, measured_signal):
    """
    Fit a 4PL using all replicates (Option A).
    Inputs:
      - true_conc: 1D array of unique concentrations, length m
      - measured_signal: either
           * 2D array shape (m, nrep) with replicates per row, OR
           * 1D array of length m * nrep (row-major ordering)

    Outputs:
        - popt: optimal parameters [bottom, top, logEC50, hill_slope]
        - pcov: covariance matrix of the parameters
        - fit_info: dictionary with keys:
            * 'residuals': array of residuals (measured - fitted)
            * 's_sq': reduced chi-squared
            * 'dof': degrees of freedom
            * 'fitted': array of fitted values
    """

    true_conc = np.asarray(true_conc, dtype=float)
    measured_signal = np.asarray(measured_signal, dtype=float)

    m = len(true_conc)

    # --- interpret measured_signal shape ---
    if measured_signal.ndim == 1:
        if measured_signal.size % m != 0:
            raise ValueError(
                "1D measured_signal length must be multiple of len(true_conc)."
            )

        nrep = measured_signal.size // m
        y = measured_signal.ravel()
        x = np.repeat(true_conc, nrep)
        measured_signal_reps = measured_signal.reshape(m, nrep)

    elif measured_signal.ndim == 2:
        if measured_signal.shape[0] != m:
            raise ValueError(
                "If measured_signal is 2D, first dimension must match len(true_conc)."
            )

        measured_signal_reps = measured_signal
        m2, nrep = measured_signal_reps.shape
        x = np.repeat(true_conc, nrep)
        y = measured_signal_reps.ravel()
    else:
        raise ValueError("measured_signal must be 1D or 2D array.")

    # Reasonable initial guesses for (bottom, top, logEC50, hill)
    bottom = float(np.min(measured_signal))
    top = float(np.max(measured_signal))
    positive = true_conc[true_conc > 0]
    logEC50 = float(np.log10(np.median(positive))) if positive.size > 0 else 0.0
    hill = 1.0
    p0 = [bottom, top, logEC50, hill]

    # --- per-concentration sample std (ddof=1) and map to per-measurement sigma ---
    sd_per_conc = np.std(measured_signal_reps, axis=1, ddof=1)  # shape (m,)

    # If any sd are zero, replace with a small floor: 10% of the smallest nonzero sd if available
    nonzero = sd_per_conc[sd_per_conc > 0]
    if nonzero.size > 0:
        floor = np.min(nonzero) * 1e-1
    else:
        # fallback if all replicates identical (rare): use an absolute floor
        floor = 1e-9
    sd_per_conc = np.where(sd_per_conc == 0, floor, sd_per_conc)

    sigma = np.repeat(sd_per_conc, nrep)  # per-measurement sigma

    # Initial Guesses (p0): For complex functions (like multiple peaks), curve_fit
    # might get "stuck." Providing an initial guess using the p0 argument helps
    # the algorithm start in the right neighborhood.

    # Bounds: You can restrict the search. For example, if you know a physical
    # constant must be positive, you can set bounds=(0, np.inf).

    # Sigma (Weighting): If some of your data points are more reliable than
    # others, you can pass a sigma array to give those points more "weight" in the calculation.

    # popt: Optimal values for the parameters.
    # pcov: The estimated covariance of popt.
    # popt, pcov = curve_fit(logistic4pl, true_conc, measured_signal, p0=p0, \
    # sigma=sigma, absolute_sigma=False, maxfev=10000)

    popt, pcov = curve_fit(
        logistic4pl, x, y, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=10000
    )

    fitted = logistic4pl(x, *popt)
    residuals = y - fitted

    # Degrees of Freedom (dof): This is the number of data points minus the number
    # of parameters you are trying to find. For a 4PL fit with 10 data points,
    # $dof = 10 - 4 = 6$. It represents how much "extra" information you have to
    # verify the fit.
    dof = len(y) - len(popt)
    if dof <= 0:
        raise ValueError(
            "Degrees of freedom <= 0. Need more data points than parameters."
        )

    # Reduced Chi-Squared: You are squaring all the errors (to make them positive),
    # summing them up, and then dividing by the degrees of freedom.
    chi2 = np.sum((residuals / sigma) ** 2)
    chi2_red = chi2 / dof

    fit_info = {"residuals": residuals, "s_sq": chi2_red, "dof": dof, "fitted": fitted}
    return popt, pcov, fit_info


def plt_4pl_curve(
    true_conc,
    measured_signal,
    popt,
    include_x_zero: bool = False,
    title_suffix="",
    show_plt: bool = True,
):
    """
    Plot the fitted 4PL curve along with the measured data points.
    Args:
        true_conc (array-like): 1D array of unique concentrations.
        measured_signal (array-like): Measured signal values (1D or 2D array
                                        with replicates).
        popt (list): Fitted parameters [bottom, top, logEC50, hill_slope].
    """

    positive = true_conc[true_conc > 0]
    xmin = positive.min()
    xmax = positive.max()

    if include_x_zero:
        x_grid = np.logspace(np.log10(xmin / 10), np.log10(xmax), 300)
    else:
        x_grid = np.logspace(np.log10(xmin), np.log10(xmax), 300)

    y_grid = logistic4pl(x_grid, *popt)

    # Prepare data for plotting means + error bars (optional but common)
    measured_signal = np.asarray(measured_signal, dtype=float)

    if measured_signal.ndim == 1:
        m = len(true_conc)
        nrep = measured_signal.size // m
        y_reps = measured_signal.reshape(m, nrep)
    else:
        y_reps = measured_signal

    y_mean = y_reps.mean(axis=1)
    y_sd = y_reps.std(axis=1, ddof=1)

    zero_x = xmin / 10

    if include_x_zero:
        # ---- Pseudo-x for zero (plotting only) ----
        plot_x = np.where(true_conc == 0, zero_x, true_conc)
    else:
        # remove zero points
        mask = true_conc > 0
        plot_x = true_conc[mask]
        y_mean = y_mean[mask]
        y_sd = y_sd[mask]

    fig, ax = plt.subplots()

    # Replicate scatter
    # for i, px in enumerate(plot_x):
    # ax.scatter([px] * y_reps.shape[1], y_reps[i], alpha=0.5)

    # Mean ± SD (optional)
    # ax.errorbar(true_conc, y_mean, yerr=y_sd, fmt="o", capsize=3, label="Mean ± SD")
    # Mean ± SD
    ax.errorbar(
        plot_x,
        y_mean,
        yerr=y_sd,
        fmt="o",
        capsize=3,
        label="Mean ± SD",
        zorder=3,
    )

    # Fitted curve
    ax.plot(
        x_grid,
        y_grid,
        "-",
        label="4PL fit",
    )

    # --- Axis formatting ---
    ax.set_xscale("log")
    right_pad = 1.2
    if include_x_zero:
        ax.set_xlim(zero_x, xmax * right_pad)
    else:
        ax.set_xlim(xmin, xmax * right_pad)

    ax.set_xlabel("Concentration")
    ax.set_ylabel("Measured signal")
    ax.set_title("Calibration curve (4PL)" + title_suffix)
    ax.legend()

    # --- Custom ticks to show zero explicitly ---
    # if include_x_zero is False:
    # xticks = [zero_x] + list(positive)
    # xticklabels = ["0"] + [f"{x:g}" for x in positive]

    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticklabels)

    # right_pad = 1.2
    # ax.set_xlim(zero_x, xmax * right_pad)

    # ---- Log ticks (let Matplotlib decide spacing) ----
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto"))

    # ---- Custom formatter to label zero ----
    def log_formatter(x, pos):
        if include_x_zero and np.isclose(x, zero_x):
            return "0"
        return f"{x:g}"

    if include_x_zero is True:
        ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))

    plt.tight_layout()

    if show_plt:
        plt.show()
    else:
        return fig, ax


def predict_4pl(x, popt):
    """
    Predict measured signal using the fitted 4PL parameters.

    Args:
        x (array-like): Concentration values to predict.
        popt (list): Fitted parameters [bottom, top, logEC50, hill_slope].

    Returns:
        array: Predicted measured signal values.
    """
    conc = np.asarray(x, dtype=float)
    return logistic4pl(conc, *popt)


def inverse_4pl(y, popt, clip: bool = True):
    """
    Inverse prediction: Given a measured signal, estimate the concentration
    using the fitted 4PL parameters.

    Args:
        y (array-like): Measured signal values.
        popt (list): Fitted parameters [bottom, top, logEC50, hill_slope].

    Returns:
        array: Estimated concentration values.
    """

    bottom, top, logEC50, hill_slope = popt
    signal = np.asarray(y, dtype=float)

    if hill_slope == 0:
        raise ValueError("hill_slope must be non-zero")

    if top == bottom:
        raise ValueError("top and bottom must be different")

    if clip:
        eps = 1e-12
        lo = min(bottom, top) + eps
        hi = max(bottom, top) - eps
        signal = np.clip(signal, lo, hi)

    # Avoid division by zero or invalid values
    ratio = (top - bottom) / (signal - bottom) - 1
    ratio = np.where(ratio <= 0, 1e-12, ratio)

    log_conc = logEC50 - (1 / hill_slope) * np.log10(ratio)
    conc = 10**log_conc
    return conc


def export_calibration_for_graphpad(
    true_conc,
    measured_signal,
    popt,
    *,
    include_x_zero=False,
    out_path=".",
    out_prefix="calibration_4pl",
    curve_points=1000,
):
    """
    Exports:
      1) {out_prefix}_observed.csv   -> conc, rep1..repN, mean, sd
      2) {out_prefix}_curve.csv      -> conc_grid, y_pred
      3) {out_prefix}.pdf/.png/.svg  -> the calibration plot

    Notes for Prism:
      - Import observed.csv into an XY table (X = conc, Y = replicates OR mean)
      - Import curve.csv into another XY table to overlay fitted curve as a line
    """

    # Check if the dir exist, if not create it
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    true_conc = np.asarray(true_conc, dtype=float)
    measured_signal = np.asarray(measured_signal, dtype=float)

    # ---- normalize replicates to shape (m, nrep) ----
    m = len(true_conc)
    if measured_signal.ndim == 1:
        if measured_signal.size % m != 0:
            raise ValueError(
                "1D measured_signal length must be multiple of len(true_conc)."
            )
        nrep = measured_signal.size // m
        y_reps = measured_signal.reshape(m, nrep)
    elif measured_signal.ndim == 2:
        if measured_signal.shape[0] != m:
            raise ValueError(
                "If measured_signal is 2D, first dim must match len(true_conc)."
            )
        y_reps = measured_signal
        nrep = y_reps.shape[1]
    else:
        raise ValueError("measured_signal must be 1D or 2D.")

    y_mean = y_reps.mean(axis=1)
    y_sd = y_reps.std(axis=1, ddof=1) if nrep > 1 else np.zeros(m)

    # ---- observed table (replicates + mean + sd) ----
    rep_cols = {f"rep{i+1}": y_reps[:, i] for i in range(nrep)}
    df_obs = pd.DataFrame(
        {
            "concentration": true_conc,
            **rep_cols,
            "mean": y_mean,
            "sd": y_sd,
        }
    )
    df_obs.to_csv(f"{out_path}/{out_prefix}_observed.csv", index=False)

    # ---- curve grid table ----
    positive = true_conc[true_conc > 0]
    if positive.size == 0:
        raise ValueError(
            "Need at least one positive concentration to build a log-spaced curve grid."
        )
    xmin = positive.min()
    xmax = positive.max()

    if include_x_zero:
        # log grid that extends one decade below xmin
        x_grid = np.logspace(np.log10(xmin / 10), np.log10(xmax), curve_points)
    else:
        x_grid = np.logspace(np.log10(xmin), np.log10(xmax), curve_points)

    y_grid = predict_4pl(x_grid, popt)

    df_curve = pd.DataFrame({"concentration": x_grid, "predicted_signal": y_grid})
    df_curve.to_csv(f"{out_path}/{out_prefix}_curve.csv", index=False)

    # ---- make & save plot ----
    fig, ax = plt_4pl_curve(
        true_conc,
        measured_signal,
        popt,
        include_x_zero=include_x_zero,
        title_suffix="",
        show_plt=False,
    )

    # Save the *current* figure (the one created by plt_4pl_curve)
    fig.savefig(f"{out_path}/{out_prefix}.svg", bbox_inches="tight")

    plt.close(fig)

    return df_obs, df_curve
