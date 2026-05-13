"""
Audio preprocessing: load, normalize, and persist processed WAV clips.

Loads waveforms with librosa, resamples to 44.1 kHz, trims multi-channel
signals to stereo when more than two channels are present, then writes a
new file under ``<project_root>/data/processed/`` with the suffix
``_processed`` inserted before the original extension. Intended to run after
validation (see ``validation.py``) so inputs are known-good WAV paths.
"""

import os

import librosa
import soundfile as sf


def process_audio(file_path):
    """
    Load an audio file, normalize sample rate and channel layout, and save a processed copy.

    Writes PCM using soundfile at 44.1 kHz regardless of the source rate after
    resampling. Output lives in ``data/processed`` relative to the repository
    root (two levels above this package).

    Args:
        file_path: Path to an audio file readable by ``librosa.load``.

    Returns:
        str: Absolute path to the written file, named
            ``<stem>_processed<original_extension>``.

    Note:
        The output directory is created implicitly by ``sf.write`` only if
        the parent path exists; ensure ``data/processed`` exists or extend
        this function if you need automatic directory creation.
    """
    y, sr = librosa.load(file_path, sr=None)

    y = normalizeSampleRate(y, sr)
    y = normalizeChannels(y)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "data", "processed")

    base_name = os.path.basename(file_path)
    root, extension = os.path.splitext(base_name)
    new_file_path = os.path.join(output_dir, root + "_processed" + extension)


    sf.write(new_file_path, y=y, samplerate=44100)
    return new_file_path


def normalizeSampleRate(y, sr):
    """
    Resample a waveform to 44.1 kHz when the native rate differs.

    Args:
        y: Time-domain samples as loaded by librosa (1-D or 2-D).
        sr: Sample rate of ``y`` in Hz.

    Returns:
        ndarray: ``y`` unchanged if ``sr`` is already 44100; otherwise the
        resampled signal at 44100 Hz.
    """
    target_sr = 44100

    if sr != target_sr:
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return y_resampled
    else:
        return y


def normalizeChannels(y):
    """
    Reduce layouts with more than two channels to the first two channels.

    For inputs with more than two channels (as inferred by ``readChannelCount``),
    assumes a shape ``(n_channels, n_samples)`` and keeps ``y[:2, :]``.
    Mono (1-D) and stereo (2-D, two rows) signals are returned unchanged.

    Args:
        y: Waveform array in librosa channel-major layout when 2-D.

    Returns:
        ndarray: Stereo pair or original mono/stereo ``y``.
    """
    channel_count = readChannelCount(y)
    if channel_count > 2:
        y_stereo = y[:2, :]
        return y_stereo
    return y


def readChannelCount(y):
    """
    Infer number of audio channels from a librosa-loaded array ``y``.

    Args:
        y: Waveform as returned by ``librosa.load``: 1-D mono ``(n_samples,)``,
            or 2-D ``(n_channels, n_samples)``. For ``ndim > 2``, the code assumes
            a batch layout ``(..., channels, samples)`` and reads ``y.shape[1]``.

    Returns:
        int: Channel count (at least 1).
    """

    if y.ndim == 1:
        channels = 1
    elif y.ndim == 2:
        channels = y.shape[0]
    else:
        channels = y.shape[1] # For 3D batches structured as (tracks, channels, samples)
    
    return channels