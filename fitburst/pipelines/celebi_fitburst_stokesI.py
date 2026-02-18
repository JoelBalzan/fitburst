#!/usr/bin/env python3

"""
CELEBI → fitburst bridge for Stokes I HTR products.

Usage (basic):
     ./celebi_fitburst_stokesI.py 250607_summary.txt

This will:
  1. Parse the CELEBI summary file to find the Stokes-I HTR dynspec and
      relevant metadata (cfreq, bw, DM, reference frequency, crop MJD).
  2. Load the Stokes-I dynspec .npy, infer the frequency axis (freq0, df)
      from cfreq, bw and the number of channels, and package a fitburst
      generic .npz.

Notes/assumptions:
  * The HTR dynamic spectrum is assumed to be already dedispersed at the
        DM listed in the summary (htr_DM). This is written into the
        fitburst file as the "full" DM, and metadata.is_dedispersed=True.
    * The time resolution dt [s] is set to the Nyquist sampling time for the
        total bandwidth, dt = 1 / (2 * bw), where bw is in Hz.
  * The start time of bin 0 (times_bin0) is taken from the summary entry
        'crop_MJD' and used as t0_mjd.
    * The default initial arrival_time guess is placed at the centre of the
        time window, relative to t0_mjd.

To run the fitburst pipeline on the resulting file, call
fitburst_pipeline.py separately, for example:

	./fitburst_pipeline.py OUTPUT_FITBURST.npz --preprocess --verbose

The produced .npz follows the fitburst "generic" schema:

metadata = {
    "bad_chans"      : # a Python list of indices corresponding to frequency
                        # channels to zero-weight
    "freqs_bin0"     : # a floating-point scalar indicating the value of
                        # frequency bin at index 0, in MHz
    "is_dedispersed" : # a boolean indicating if spectrum is already
                        # dedispersed (True) or not (False)
    "num_freq"       : # an integer scalar indicating the number of
                        # frequency bins/channels
    "num_time"       : # an integer scalar indicating the number of time
                        # bins
    "times_bin0"     : # a floating-point scalar indicating the value of
                        # time bin at index 0, in MJD
    "res_freq"       : # a floating-point scalar indicating the frequency
                        # resolution, in MHz
    "res_time"       : # a floating-point scalar indicating the time
                        # resolution, in seconds
}

burst_parameters = {
    "amplitude"            : # a list containing the log (base 10) of the
                               # overall signal amplitude
    "arrival_time"         : # a list containing the arrival times, in
                               # seconds, relative to times_bin0
    "burst_width"          : # a list containing the temporal widths, in
                               # seconds
    "dm"                   : # a list containing the dispersion measures
                               # (DM), in parsec per cubic centimeter
    "dm_index"             : # a list containing the exponents of frequency
                               # dependence in DM delay
    "ref_freq"             : # a list containing the reference frequencies
                               # for arrival-time and power-law parameter
                               # estimates, in MHz (held fixed)
    "scattering_index"     : # a list containing the exponents of frequency
                               # dependence in scatter-broadening
    "scattering_timescale" : # a list containing the scattering
                               # timescales, in seconds
    "spectral_index"       : # a list containing the power-law spectral
                               # indices
    "spectral_running"     : # a list containing the power-law spectral
                               # running
}

"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


def parse_summary(path: str) -> Dict[str, str]:
    """Parse a CELEBI-style summary text file into a flat dict.

    Lines are expected in the form:
        key:  value   [units]
    or
        key:  value

    Keys are normalised to a compact form without surrounding whitespace.
    """
    meta: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            key = key.strip()
            # Strip comments/units in brackets from the value.
            # e.g. "919.5                                                       [MHz]"
            rest = rest.strip()
            # Remove inline comments / units in square brackets.
            # Keep only the leading non-whitespace token.
            if rest:
                # Split on whitespace and take first token that looks like a number or path.
                # For paths with spaces (unlikely here), fall back to full rest.
                tokens = rest.split()
                if len(tokens) == 1:
                    value = tokens[0]
                else:
                    # Heuristic: for dsI, dsQ, etc., keep full rest (path); otherwise first token.
                    if key.lower().startswith("ds"):
                        value = rest
                    else:
                        value = tokens[0]
            else:
                value = ""
            meta[key] = value
    return meta


def infer_freq_axis(cfreq_mhz: float, bw_mhz: float, num_chan: int) -> Tuple[float, float]:
    """Infer (freq0, df) from cfreq, total bandwidth and number of channels.

    Assumes equally spaced channels that tile the band. We treat cfreq as the
    band centre and return:
        df    = bw / N
        freq0 = cfreq - bw/2 + df/2   (centre of channel 0)
    """
    df = bw_mhz / float(num_chan)
    freq0 = cfreq_mhz - 0.5 * bw_mhz + 0.5 * df
    return freq0, df


def measure_spectral_parameters(
    data: np.ndarray,
    peak_idx: int,
    freqs_mhz: np.ndarray,
    ref_freq_mhz: float,
    time_window: int = 5,
    min_chan_snr: float = 2.0,
) -> Tuple[float, float]:
    """Estimate (spectral_index, spectral_running) at a given peak time.

    Uses a simple quadratic fit in log-space:
        ln S(f) = c0 + alpha * ln(f/f_ref) + beta * ln(f/f_ref)^2

    where alpha is the spectral index and beta is the spectral running.

    The spectrum is taken around the peak time. For very narrow pulses we
    use the per-channel maximum in the window (to avoid diluting the signal);
    for broader pulses we use a per-channel mean.

    A per-channel baseline and noise are estimated robustly across the full
    time axis (median and MAD). Channels are selected based on a simple
    per-channel SNR threshold and the fit is noise-weighted.

    Returns (0.0, 0.0) if a fit cannot be performed.
    """
    if data.ndim != 2:
        return 0.0, 0.0

    num_time = data.shape[1]
    if num_time < 3:
        return 0.0, 0.0

    if time_window < 1:
        time_window = 1

    peak_idx = int(peak_idx)
    if peak_idx < 0 or peak_idx >= num_time:
        return 0.0, 0.0

    half_window = time_window // 2
    t_start = max(0, peak_idx - half_window)
    t_end = min(num_time, peak_idx + half_window + 1)

    # Robust per-channel baseline/noise across all times.
    baseline_spec = np.nanmedian(data, axis=1)
    mad_spec = np.nanmedian(np.abs(data - baseline_spec[:, None]), axis=1)
    noise_spec = 1.4826 * mad_spec
    # Fallback if MAD is degenerate.
    bad_noise = ~np.isfinite(noise_spec) | (noise_spec <= 0)
    if np.any(bad_noise):
        noise_spec = noise_spec.copy()
        noise_spec[bad_noise] = np.nanstd(data[bad_noise, :], axis=1)

    window = data[:, t_start:t_end]
    # Use max for narrow windows, mean otherwise.
    if (t_end - t_start) <= 3:
        spec_raw = np.nanmax(window, axis=1)
    else:
        spec_raw = np.nanmean(window, axis=1)

    spectrum = spec_raw - baseline_spec
    snr = spectrum / (noise_spec + 1e-30)

    valid_mask = (
        np.isfinite(freqs_mhz)
        & np.isfinite(spectrum)
        & np.isfinite(snr)
        & (freqs_mhz > 0)
        & (spectrum > 0)
        & (snr >= float(min_chan_snr))
    )
    n_valid = int(np.count_nonzero(valid_mask))
    if n_valid < 3:
        return 0.0, 0.0

    x = np.log(freqs_mhz[valid_mask] / float(ref_freq_mhz))
    y = np.log(spectrum[valid_mask])
    w = np.clip(snr[valid_mask], 0.0, 50.0)

    # Guard against a degenerate design (all x identical, etc.).
    if not np.isfinite(x).all() or np.nanstd(x) <= 0:
        return 0.0, 0.0

    try:
        if n_valid >= 5:
            # np.polyfit returns [c2, c1, c0] for deg=2: y = c2*x^2 + c1*x + c0
            c2, c1, _c0 = np.polyfit(x, y, deg=2, w=w)
            spectral_running = float(c2)
            spectral_index = float(c1)
        else:
            # Too few points for a stable quadratic; fall back to pure power-law.
            c1, _c0 = np.polyfit(x, y, deg=1, w=w)
            spectral_running = 0.0
            spectral_index = float(c1)
    except Exception:
        return 0.0, 0.0

    # Keep guesses in a reasonable range to avoid pathological starts.
    if not np.isfinite(spectral_index) or not (-10.0 <= spectral_index <= 10.0):
        spectral_index = 0.0
    if not np.isfinite(spectral_running) or not (-50.0 <= spectral_running <= 50.0):
        spectral_running = 0.0

    return spectral_index, spectral_running


def write_diagnostic_plot(
    *,
    data: np.ndarray,
    dt: float,
    freqs_mhz: np.ndarray,
    arrival_times_s: List[float],
    output_path: str,
    title: Optional[str] = None,
) -> None:
    """Write a diagnostic plot: pulse profile + dynamic spectrum with labeled peaks."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[CELEBI→fitburst] Diagnostic plot skipped (matplotlib unavailable): {e}")
        return

    if data.ndim != 2:
        print("[CELEBI→fitburst] Diagnostic plot skipped (data not 2D)")
        return

    num_freq, num_time = data.shape
    if num_time < 2 or num_freq < 2:
        print("[CELEBI→fitburst] Diagnostic plot skipped (data too small)")
        return

    times_s = np.arange(num_time) * float(dt)
    profile = np.nanmean(data, axis=0)

    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.0])

    ax_top = fig.add_subplot(gs[0, 0])
    ax_ds = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # Top: pulse profile
    ax_top.plot(times_s, profile, lw=1.0, color="k")
    ax_top.set_ylabel("Profile (mean over freq)")
    ax_top.grid(True, alpha=0.2)

    # Bottom: dynamic spectrum
    # Choose robust scaling to make structure visible.
    vmin, vmax = np.nanpercentile(data, [5, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = None, None

    extent = [times_s[0], times_s[-1], float(freqs_mhz[0]), float(freqs_mhz[-1])]
    im = ax_ds.imshow(
        data,
        aspect="auto",
        origin="upper",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_ds.set_xlabel("Time since bin0 (s)")
    ax_ds.set_ylabel("Frequency (MHz)")
    cbar = fig.colorbar(im, ax=ax_ds, pad=0.01)
    cbar.set_label("Intensity")

    # Mark and label peaks
    y_text = np.nanmax(profile)
    if not np.isfinite(y_text):
        y_text = 0.0

    for i, t_s in enumerate(arrival_times_s, start=1):
        ax_top.axvline(t_s, color="C1", lw=1.0, alpha=0.8)
        ax_ds.axvline(t_s, color="C1", lw=1.0, alpha=0.8)
        ax_top.text(
            t_s,
            y_text,
            f"{i}",
            color="C1",
            fontsize=9,
            ha="center",
            va="bottom",
        )

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle("fitburst diagnostic: detected components")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[CELEBI→fitburst] Wrote diagnostic plot: {output_path}")


def measure_scattering_timescale(
    time_series: np.ndarray,
    peak_idx: int,
    dt: float,
    baseline: float,
) -> float:
    """Estimate scattering timescale from the decay tail of a pulse.

    Fits an exponential decay to the trailing edge after the peak.

    Parameters
    ----------
    time_series : np.ndarray
        1D time series
    peak_idx : int
        Index of the peak
    dt : float
        Time resolution in seconds
    baseline : float
        Baseline level for the pulse

    Returns
    -------
    scattering_timescale : float
        Scattering timescale in seconds (0 if cannot be estimated)
    """
    # Extract trailing edge after peak
    if peak_idx >= len(time_series) - 3:
        return 0.0
    
    trail = time_series[peak_idx:]
    peak_val = trail[0]
    
    # Need significant amplitude above baseline
    if peak_val <= baseline or (peak_val - baseline) < 1e-10:
        return 0.0
    
    # Find where signal drops to baseline + (peak-baseline)/e
    target_val = baseline + (peak_val - baseline) / np.e
    
    # Find where we cross the target value
    tau_idx = 0
    for i in range(1, len(trail)):
        if trail[i] <= target_val:
            tau_idx = i
            break
    
    if tau_idx == 0:
        # Decay is slower than window, estimate from last 30% of trail
        end_idx = min(len(trail), int(0.3 * len(trail)) + 5)
        if end_idx < 3:
            return 0.0
        trail_subset = trail[:end_idx]
        # Simple exponential estimate
        try:
            log_vals = np.log(np.maximum(trail_subset - baseline, 1e-20))
            time_vals = np.arange(len(trail_subset)) * dt
            # Fit line to log: log(y) = log(A) - t/tau
            coeffs = np.polyfit(time_vals, log_vals, 1)
            tau = -1.0 / coeffs[0] if coeffs[0] < 0 else 0.0
            return max(0.0, float(tau))
        except:
            return 0.0
    
    # Estimate tau from crossing point
    tau = tau_idx * dt
    return float(tau)


def find_peak_arrival_times(
    data: np.ndarray, 
    dt: float,
    freqs_mhz: np.ndarray,
    ref_freq_mhz: float,
    cfreq_mhz: float,
    min_peak_snr: float = 6.0,
    max_peaks: int = 5,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Find significant peaks in a dynamic spectrum.

    Collapses the dynspec along the frequency axis (ignoring NaNs) and finds
    significant peaks above noise. Measures width (sigma) and scattering timescale for each.

    Parameters
    ----------
    data : np.ndarray
        2D dynamic spectrum (num_freq, num_time)
    dt : float
        Time resolution in seconds
    freqs_mhz: np.ndarray
        Frequency axis in MHz
    ref_freq_mhz: float
        Reference frequency in MHz
    cfreq_mhz: float
        Center frequency in MHz (for scaling scattering timescale)
    min_peak_snr : float, optional
        Minimum signal-to-noise ratio for a peak to be considered significant
    max_peaks : int, optional
        Maximum number of peaks to return

    Returns
    -------
    arrival_times : List[float]
        List with arrival times in seconds (relative to bin 0) for each peak
    peak_amplitudes : List[float]
        List with log10 amplitude for each peak
    burst_widths : List[float]
        List with sigma (standard deviation) width in seconds for each peak
    scattering_timescales : List[float]
        List with scattering timescale in seconds for each peak
    spectral_indices : List[float]
        List with spectral index for each peak
    spectral_runnings : List[float]
        List with spectral running for each peak
    """
    # Collapse to 1D time series by averaging over frequency (ignoring NaNs)
    time_series = np.nanmean(data, axis=0)
    
    # Handle all-NaN case
    if np.all(np.isnan(time_series)):
        # Fallback to single peak at center
        peak_idx = len(time_series) // 2
        arrival_times = [peak_idx * dt]
        peak_amplitudes = [float(np.log10(np.nanmax(data)))]
        burst_widths = [1e-3]  # Default 1 ms
        scattering_timescales = [0.0]
        spectral_indices = [0.0]
        spectral_runnings = [0.0]
        return (
            arrival_times,
            peak_amplitudes,
            burst_widths,
            scattering_timescales,
            spectral_indices,
            spectral_runnings,
        )
    
    # Calculate noise statistics from off-pulse region
    # Use first 5% of the time series as off-pulse baseline
    edge_size = max(10, int(len(time_series) * 0.05))
    off_pulse = time_series[:edge_size]
    
    # Use robust statistics for noise estimation
    noise_median = np.nanmedian(off_pulse)
    # Median absolute deviation (MAD) is more robust than std
    mad = np.nanmedian(np.abs(off_pulse - noise_median))
    noise_std = 1.4826 * mad  # Convert MAD to equivalent std for Gaussian
    
    # Fallback to standard deviation if MAD is too small
    if noise_std < 1e-20:
        noise_std = np.nanstd(off_pulse)
    
    noise_mean = noise_median
    
    # Find the global range for fallback
    ts_max = np.nanmax(time_series)
    ts_min = np.nanmin(time_series)
    
    if ts_max == ts_min or noise_std == 0:
        # Flat signal, return single peak at maximum
        peak_idx = np.nanargmax(time_series)
        arrival_times = [peak_idx * dt]
        peak_amplitudes = [float(np.log10(np.nanmax(data)))]
        burst_widths = [1e-3]  # Default 1 ms
        scattering_timescales = [0.0]
        spectral_indices = [0.0]
        spectral_runnings = [0.0]
        return (
            arrival_times,
            peak_amplitudes,
            burst_widths,
            scattering_timescales,
            spectral_indices,
            spectral_runnings,
        )
    
    # Set thresholds for peak detection
    height_threshold = noise_mean + min_peak_snr * noise_std
    prominence_threshold = min_peak_snr * noise_std
    
    # Find peaks using scipy's find_peaks with prominence and height requirements
    # Allow narrow peaks (width=1) since height and prominence thresholds filter noise
    peak_indices, properties = find_peaks(
        time_series,
        height=height_threshold,
        prominence=prominence_threshold,
        width=10,  # Allow single-sample peaks for extremely narrow bursts
        distance=20  # Minimum separation between peaks (reduced for narrow features)
    )
    
    if len(peak_indices) == 0:
        # No significant peaks found, return the global maximum
        peak_idx = np.nanargmax(time_series)
        arrival_times = [peak_idx * dt]
        peak_amplitudes = [float(np.log10(np.nanmax(data)))]
        burst_widths = [1e-3]
        scattering_timescales = [0.0]
        spectral_indices = [0.0]
        spectral_runnings = [0.0]
        print(f"[Peak finder] No peaks above SNR={min_peak_snr:.1f}, using global max")
        return (
            arrival_times,
            peak_amplitudes,
            burst_widths,
            scattering_timescales,
            spectral_indices,
            spectral_runnings,
        )
    
    # Sort peaks by height (descending) and limit to max_peaks
    peak_heights = time_series[peak_indices]
    sorted_indices = np.argsort(peak_heights)[::-1][:max_peaks]
    peak_indices = peak_indices[sorted_indices]
    
    # Sort by arrival time for output
    peak_indices = np.sort(peak_indices)
    
    arrival_times = []
    peak_amplitudes = []
    burst_widths = []
    scattering_timescales = []
    spectral_indices = []
    spectral_runnings = []
    
    # Process each peak
    for peak_idx in peak_indices:
        # Arrival time
        arrival_time = peak_idx * dt
        arrival_times.append(float(arrival_time))
        
        # Amplitude at this time slice (max over frequency)
        amp = np.nanmax(data[:, peak_idx])
        if amp <= 0 or not np.isfinite(amp):
            amp = time_series[peak_idx]
        peak_amplitudes.append(float(np.log10(max(amp, 1e-20))))
        
        # Measure width as sigma (convert FWHM to sigma by dividing by 2.355)
        peak_val = time_series[peak_idx]
        half_max = noise_mean + (peak_val - noise_mean) / 2.0
        
        # Search left from peak
        left_idx = peak_idx
        while left_idx > 0 and time_series[left_idx] > half_max:
            left_idx -= 1
        
        # Search right from peak
        right_idx = peak_idx
        while right_idx < len(time_series) - 1 and time_series[right_idx] > half_max:
            right_idx += 1
        
        # Calculate FWHM and convert to sigma
        fwhm_bins = max(2, right_idx - left_idx)
        fwhm_time = fwhm_bins * dt
        sigma = fwhm_time / 2.355  # Convert FWHM to sigma
        burst_widths.append(float(sigma))
        
        # Measure scattering timescale from frequency-collapsed time series
        # This is measured at the effective center frequency, so scale to reference frequency
        scattering_tau_cfreq = measure_scattering_timescale(time_series, peak_idx, dt, noise_mean)
        # Scale to reference frequency: tau_ref = tau_cfreq * (ref_freq / cfreq)^(-4)
        scattering_index = -4.0
        scattering_tau_ref = scattering_tau_cfreq * (ref_freq_mhz / cfreq_mhz) ** scattering_index
        scattering_timescales.append(float(scattering_tau_ref))

        # Spectral index / running from the frequency spectrum at this time.
        # Use an adaptive spectral window size. For very narrow peaks, use
        # a minimal window so the signal isn't diluted.
        spec_bins = max(1, right_idx - left_idx)
        spec_bins = int(min(25, spec_bins))
        if spec_bins % 2 == 0:
            spec_bins += 1

        spec_idx, spec_run = measure_spectral_parameters(
            data=data,
            peak_idx=peak_idx,
            freqs_mhz=freqs_mhz,
            ref_freq_mhz=ref_freq_mhz,
            time_window=spec_bins,
        )
        spectral_indices.append(float(spec_idx))
        spectral_runnings.append(float(spec_run))
    
    print(f"[Peak finder] Found {len(peak_indices)} significant peak(s) above SNR={min_peak_snr:.1f}")
    
    return (
        arrival_times,
        peak_amplitudes,
        burst_widths,
        scattering_timescales,
        spectral_indices,
        spectral_runnings,
    )


def build_fitburst_npz_from_summary(
    summary_path: str,
    output_dir: Optional[str] = None,
    arrival_time: Optional[float] = None,
    override_dm: Optional[float] = None,
    override_ref_freq: Optional[float] = None,
    input_npy: Optional[str] = None,
    override_dt: Optional[float] = None,
    override_df: Optional[float] = None,
    override_scattering: Optional[float] = None,
    min_peak_snr: float = 6.0,
    max_peaks: int = 5,
    diagnostic_plot: bool = False,
) -> str:
    """Create a fitburst-compatible .npz for Stokes I using the summary.

    Returns the path to the generated .npz file.
    """
    print(f"[CELEBI→fitburst] Parsing CELEBI summary: {summary_path}")
    meta = parse_summary(summary_path)

    # FRB name is used for default naming of some products.
    frb_name = meta.get("FRB name", "FRB").strip()

    # DM from summary (htr_DM) is used both for modelling (unless overridden)
    # and for constructing the expected cropped dynspec filename.
    try:
        htr_dm_val = float(meta["htr_DM"])
    except KeyError as e:
        raise RuntimeError("Summary missing 'htr_DM' value") from e
    except ValueError as e:
        raise RuntimeError("Could not parse 'htr_DM' value as float") from e

    # Determine input .npy file path: use provided path if given, otherwise
    # derive from summary. Use the dsI entry only to locate the parent HTR
    # directory, then point to the cropped dynspec in the "crops" subdirectory
    # with CELEBI's naming scheme, e.g.: /.../htr/crops/<FRB>_<DM>_dsI_crop.npy
    if input_npy is not None:
        dsI_path = os.path.abspath(input_npy)
        if not os.path.exists(dsI_path):
            raise FileNotFoundError(f"Provided input .npy file not found: {dsI_path}")
        print(f"[CELEBI→fitburst] Using provided input .npy: {dsI_path}")
    else:
        try:
            dsI_raw = meta["dsI"].split()[0]
        except KeyError as e:
            raise RuntimeError("Could not find 'dsI' entry in summary file") from e

        if not os.path.isabs(dsI_raw):
            # Interpret relative to summary file location.
            dsI_raw = os.path.join(os.path.dirname(os.path.abspath(summary_path)), dsI_raw)

        htr_dir = os.path.dirname(dsI_raw)
        crops_dir = os.path.join(htr_dir, "crops")
        crop_name = f"{frb_name}_{htr_dm_val:.3f}_dsI_crop.npy"
        dsI_path = os.path.join(crops_dir, crop_name)

        if not os.path.exists(dsI_path):
            raise FileNotFoundError(f"Stokes-I cropped dynspec not found: {dsI_path}")
        print(f"[CELEBI→fitburst] Using Stokes-I cropped dynspec: {dsI_path}")

    # Frequencies and bandwidth.
    try:
        cfreq_mhz = float(meta["cfreq"])
        bw_mhz = float(meta["bw"])
    except KeyError as e:
        raise RuntimeError("Summary missing 'cfreq' or 'bw' entry") from e
    print(f"[CELEBI→fitburst] cfreq = {cfreq_mhz:.3f} MHz, bw = {bw_mhz:.3f} MHz")

    # DM for modelling: use htr_DM from the summary unless overridden.
    dm_val = htr_dm_val
    if override_dm is not None:
        dm_val = override_dm
    print(f"[CELEBI→fitburst] DM (htr_DM / override) = {dm_val:.6f} pc/cm^3")

    # Reference frequency: prefer DM_ref_freq, then corr_ref_freq, then cfreq.
    ref_freq_mhz: float
    for key in ("DM_ref_freq", "corr_ref_freq", "cfreq"):
        if key in meta:
            try:
                ref_freq_mhz = float(meta[key])
                break
            except ValueError:
                continue
    else:
        raise RuntimeError("Summary missing DM/ref frequency entries")

    if override_ref_freq is not None:
        ref_freq_mhz = override_ref_freq
    print(f"[CELEBI→fitburst] Reference frequency = {ref_freq_mhz:.3f} MHz")

    # Time of first bin in MJD: use crop_MJD if present, else corr_MJD if present.
    t0_mjd: Optional[float] = None
    for key in ("crop_MJD", "corr_MJD"):
        if key in meta:
            try:
                t0_mjd = float(meta[key])
                break
            except ValueError:
                continue

    # Load dynamic spectrum to determine num_freq and num_time.
    data_full = np.load(dsI_path)
    if data_full.ndim != 2:
        raise ValueError(f"Expected 2D dynspec in {dsI_path}, got shape {data_full.shape}")

    num_freq, num_time = data_full.shape
    print(f"[CELEBI→fitburst] Loaded dynspec with shape (num_freq={num_freq}, num_time={num_time})")
    
    # Frequency resolution: use override if provided, otherwise infer from bandwidth
    if override_df is not None:
        df_mhz = override_df
        # Compute freq0 assuming df and cfreq
        freq0_mhz = cfreq_mhz - 0.5 * bw_mhz + 0.5 * df_mhz
        print(f"[CELEBI→fitburst] Using override frequency resolution df = {df_mhz:.6f} MHz")
    else:
        freq0_mhz, df_mhz = infer_freq_axis(cfreq_mhz, bw_mhz, num_freq)
        print(f"[CELEBI→fitburst] Computed frequency resolution df = {df_mhz:.6f} MHz")

    # Time resolution: use override if provided, otherwise compute Nyquist sampling
    # time for total bandwidth. bw_mhz is the total bandwidth in MHz; convert to
    # Hz for dt. dt = 1 / (2 * bw_Hz).
    if override_dt is not None:
        dt = override_dt
        print(f"[CELEBI→fitburst] Using override time resolution dt = {dt:.6e} s")
    else:
        bw_hz = bw_mhz * 1.0e6
        dt = 1.0 / (2.0 * bw_hz)
        print(f"[CELEBI→fitburst] Computed Nyquist dt = {dt:.6e} s")

    # Decide on output path.
    default_name = os.path.splitext(os.path.basename(dsI_path))[0] + "_fitburst.npz"
    if output_dir is None:
        # Default: current working directory.
        outdir = os.getcwd()
        output_npz = os.path.join(outdir, default_name)
    else:
        # If the argument looks like a .npz file, treat it as a full
        # output filename; otherwise as a directory containing the
        # default name. A value of "." means the current working
        # directory.
        candidate = os.path.abspath(output_dir)
        if candidate.lower().endswith(".npz"):
            outdir = os.path.dirname(candidate) or os.getcwd()
            output_npz = candidate
        else:
            outdir = candidate
            output_npz = os.path.join(outdir, default_name)

    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)

    # Frequency axis for spectral fitting (MHz).
    freqs_mhz = freq0_mhz + np.arange(num_freq) * df_mhz

    # If no arrival_time is supplied, find the main peak in the dynspec.
    if arrival_time is None:
        (
            arrival_times,
            peak_amplitudes,
            burst_widths,
            scattering_timescales,
            spectral_indices,
            spectral_runnings,
        ) = find_peak_arrival_times(
            data_full,
            dt,
            freqs_mhz,
            ref_freq_mhz,
            cfreq_mhz,
            min_peak_snr=min_peak_snr,
            max_peaks=max_peaks,
        )
        num_components = len(arrival_times)
        
        for i, (arr_t, amp, width, scat, spec_idx, spec_run) in enumerate(
            zip(
                arrival_times,
                peak_amplitudes,
                burst_widths,
                scattering_timescales,
                spectral_indices,
                spectral_runnings,
            )
        ):
            peak_bin = int(arr_t / dt)
            print(
                f"[CELEBI→fitburst] Peak {i+1}: t={arr_t:.6e} s (bin {peak_bin}), "
                f"log10(amp)={amp:.3f}, width={width:.6e} s, scattering={scat:.6e} s, "
                f"spectral_index={spec_idx:.3f}, spectral_running={spec_run:.3f}"
            )
    else:
        # Single component with user-specified arrival time
        arrival_times = [float(arrival_time)]
        peak_amplitudes = [float(np.log10(np.nanmax(data_full)))]
        burst_widths = [1e-3]  # Default 1 ms
        scattering_timescales = [0.0]
        peak_idx = int(np.clip(int(round(arrival_times[0] / dt)), 0, num_time - 1))
        spec_idx, spec_run = measure_spectral_parameters(
            data=data_full,
            peak_idx=peak_idx,
            freqs_mhz=freqs_mhz,
            ref_freq_mhz=ref_freq_mhz,
        )
        spectral_indices = [float(spec_idx)]
        spectral_runnings = [float(spec_run)]
        num_components = 1
        print(f"[CELEBI→fitburst] Using user-specified arrival_time = {arrival_times[0]:.6e} s")

    if diagnostic_plot:
        diag_path = os.path.splitext(output_npz)[0] + "_diagnostic.png"
        write_diagnostic_plot(
            data=data_full,
            dt=dt,
            freqs_mhz=freqs_mhz,
            arrival_times_s=arrival_times,
            output_path=diag_path,
            title=os.path.basename(output_npz),
        )

    # Override scattering timescale if specified
    if override_scattering is not None:
        scattering_timescales = [float(override_scattering)] * num_components
        print(f"[CELEBI→fitburst] Using override scattering timescale = {override_scattering:.6e} s for all components")

    # Identify bad frequency channels as those containing any NaNs.
    bad_chan_mask = np.any(np.isnan(data_full), axis=1)
    bad_chans = np.where(bad_chan_mask)[0].tolist()
    if bad_chans:
        print(
            f"[CELEBI→fitburst] Found {len(bad_chans)} bad frequency channels "
            f"with NaNs: {bad_chans}"
        )
    else:
        print("[CELEBI→fitburst] No NaNs found; no bad frequency channels flagged")

    # Prepare metadata and burst parameters following stokesI_to_fitburst_npz.py logic.
    metadata = {
        "bad_chans": bad_chans,
        "freqs_bin0": float(freq0_mhz),
        "is_dedispersed": True,  # CELEBI HTR dynspec is assumed dedispersed.
        "num_freq": int(num_freq),
        "num_time": int(num_time),
        "times_bin0": float(t0_mjd) if t0_mjd is not None else float("nan"),
        "res_freq": float(df_mhz),
        "res_time": float(dt),
    }

    burst_parameters = {
        "amplitude": peak_amplitudes,
        "arrival_time": arrival_times,
        "burst_width": burst_widths,
        "dm": [float(dm_val)] * num_components,
        "dm_index": [-2.0] * num_components,
        "ref_freq": [float(ref_freq_mhz)] * num_components,
        "scattering_index": [-4.0] * num_components,
        "scattering_timescale": scattering_timescales,
        "spectral_index": spectral_indices,
        "spectral_running": spectral_runnings,
    }
    print("[CELEBI→fitburst] Built metadata and burst_parameters dictionaries")

    np.savez(
        output_npz,
        data_full=data_full[::-1, :],
        metadata=metadata,
        burst_parameters=burst_parameters,
    )

    print(f"[CELEBI→fitburst] Wrote fitburst npz: {output_npz}")
    print(f"  data_full shape = {data_full.shape}")
    print(f"  metadata: {metadata}")
    print(f"  burst_parameters: { {k: (v if isinstance(v, list) else float(v)) for k, v in burst_parameters.items()} }")
    print(f"  num_freq = {num_freq}, num_time = {num_time}")
    print(f"  freq0 = {freq0_mhz:.6f} MHz, df = {df_mhz:.6f} MHz, dt = {dt:.6e} s")
    print(f"  DM = {dm_val:.6f} pc/cm^3, ref_freq = {ref_freq_mhz:.3f} MHz")
    if t0_mjd is not None:
        print(f"  t0_mjd = {t0_mjd:.12f} MJD")

    return output_npz


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fitburst npz for Stokes I from a CELEBI summary. "
            "Run fitburst_pipeline.py on the output file in a separate step."
        )
    )
    parser.add_argument(
        "summary",
        help="Path to CELEBI summary text file (e.g. 250607_summary.txt)",
    )
    parser.add_argument(
        "--input-npy",
        type=str,
        default=None,
        help=(
            "Path to input .npy dynamic spectrum file. If not provided, "
            "the path will be derived from the summary file's dsI entry "
            "and htr_DM value using CELEBI's naming convention."
        ),
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help=(
            "Optional output location for the fitburst .npz. "
            "If this ends with .npz, it is treated as the full output "
            "filename. Otherwise it is treated as an output directory "
            "containing <dsI_basename>_fitburst.npz. If omitted, defaults "
            "to the summary directory; use '.' for the current working "
            "directory."
        ),
    )
    parser.add_argument(
        "--arrival-time",
        type=float,
        default=None,
        help=(
            "Initial arrival_time guess relative to t0_mjd [s]. "
            "Defaults to the centre of the time window if not set."
        ),
    )
    parser.add_argument(
        "--dm",
        type=float,
        default=None,
        help="Override DM used in fitburst file [pc/cm^3].",
    )
    parser.add_argument(
        "--ref-freq",
        type=float,
        default=None,
        help=(
            "Override reference frequency used in fitburst file [MHz]. "
            "Defaults to DM_ref_freq, corr_ref_freq, or cfreq from the summary."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help=(
            "Override time resolution [seconds]. If not provided, it will be "
            "computed as the Nyquist sampling time: dt = 1 / (2 * bandwidth)."
        ),
    )
    parser.add_argument(
        "--df",
        type=float,
        default=None,
        help=(
            "Override frequency resolution [MHz]. If not provided, it will be "
            "computed as: df = bandwidth / num_channels."
        ),
    )
    parser.add_argument(
        "--scattering",
        type=float,
        default=None,
        help=(
            "Override scattering timescale [seconds] for all burst components. "
            "If not provided, it will be measured from the pulse decay."
        ),
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=6.0,
        help=(
            "Minimum signal-to-noise ratio for peak detection. Peaks must be "
            "at least this many sigma above the off-pulse noise (first 5%% of "
            "the time window). Default: 6.0."
        ),
    )
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=5,
        help=(
            "Maximum number of peaks to detect in the dynamic spectrum. "
            "Default: 5."
        ),
    )

    parser.add_argument(
        "--diagnostic-plot",
        action="store_true",
        help=(
            "Write a diagnostic PNG next to the output .npz showing the dynamic "
            "spectrum and the pulse profile with detected components labeled."
        ),
    )

    args = parser.parse_args(argv)

    npz_path = build_fitburst_npz_from_summary(
        summary_path=args.summary,
        output_dir=args.output_npz,
        arrival_time=args.arrival_time,
        override_dm=args.dm,
        override_ref_freq=args.ref_freq,
        input_npy=args.input_npy,
        override_dt=args.dt,
        override_df=args.df,
        override_scattering=args.scattering,
        min_peak_snr=args.min_snr,
        max_peaks=args.max_peaks,
        diagnostic_plot=args.diagnostic_plot,
    )
    # Nothing else to do; pipeline should be run separately.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
