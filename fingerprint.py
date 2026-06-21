"""
Core fingerprinting logic for the Q3 song identifier.

Same algorithm used in the Q3A notebook (spectrogram -> constellation peaks ->
paired hashes -> offset-histogram matching), just pulled into a module so both
the indexing script and the Streamlit app can import it.
"""
import os
import pickle
from collections import defaultdict, Counter

import numpy as np
import librosa
from scipy.ndimage import maximum_filter

SR = 11025
N_FFT = 4096
HOP = 2048

AMP_MIN = -42
NBHD = (20, 20)

FAN_OUT = 5
DT_MIN = 1
DT_MAX = 100


def load_audio(path_or_buffer, sr=SR, duration=None, offset=0.0):
    y, _ = librosa.load(path_or_buffer, sr=sr, mono=True, duration=duration, offset=offset)
    return y


def spectrogram_db(y, n_fft=N_FFT, hop=HOP):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann"))
    return librosa.amplitude_to_db(S, ref=np.max)


def get_peaks(S_db, amp_min=AMP_MIN, nbhd=NBHD):
    local_max = maximum_filter(S_db, size=nbhd) == S_db
    loud_enough = S_db > amp_min
    freq_idx, time_idx = np.where(local_max & loud_enough)
    return list(zip(time_idx, freq_idx))  # (time_bin, freq_bin)


def make_hashes(peaks, fan_out=FAN_OUT, dt_min=DT_MIN, dt_max=DT_MAX):
    peaks = sorted(peaks)
    out = []
    for i, (t1, f1) in enumerate(peaks):
        for j in range(1, fan_out + 1):
            if i + j >= len(peaks):
                break
            t2, f2 = peaks[i + j]
            dt = t2 - t1
            if dt_min <= dt <= dt_max:
                out.append(((f1, f2, dt), t1))
    return out


def fingerprint_signal(y):
    """returns (S_db, peaks, hashes) for an already-loaded mono signal"""
    S_db = spectrogram_db(y)
    peaks = get_peaks(S_db)
    hashes = make_hashes(peaks)
    return S_db, peaks, hashes


def fingerprint_file(path, duration=None, offset=0.0):
    y = load_audio(path, duration=duration, offset=offset)
    return fingerprint_signal(y)


# ---------------------------------------------------------------------------
# database build / load / save
# ---------------------------------------------------------------------------

def build_database(song_files):
    """song_files: list of paths. returns (database dict, song_names list)"""
    database = defaultdict(list)
    song_names = []
    for sid, path in enumerate(song_files):
        name = os.path.splitext(os.path.basename(path))[0]
        song_names.append(name)
        _, _, hashes = fingerprint_file(path)
        for h, t1 in hashes:
            database[h].append((sid, t1))
    return dict(database), song_names


def save_database(database, song_names, out_path="database.pkl"):
    with open(out_path, "wb") as f:
        pickle.dump({"database": database, "song_names": song_names}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_database(path="database.pkl"):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["database"], d["song_names"]


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def match_query(peaks, database, top_k=5):
    """peaks: list of (time_bin, freq_bin) from the query clip.
    returns (sorted list of (votes, song_id, offset), the offset-vote breakdown for plotting)"""
    hashes = make_hashes(peaks)
    offset_votes = defaultdict(Counter)
    for h, t_query in hashes:
        for sid, t_db in database.get(h, []):
            offset_votes[sid][t_db - t_query] += 1

    scores = []
    for sid, counter in offset_votes.items():
        best_offset, count = counter.most_common(1)[0]
        scores.append((count, sid, best_offset))
    scores.sort(reverse=True)
    return scores[:top_k], offset_votes


def identify(path_or_buffer, database, song_names, duration=None):
    """convenience wrapper used by both app modes. returns a dict of everything
    the single-clip UI needs to display."""
    y = load_audio(path_or_buffer, duration=duration)
    S_db, peaks, _ = fingerprint_signal(y)
    results, offset_votes = match_query(peaks, database)

    if not results:
        return {
            "prediction": None, "S_db": S_db, "peaks": peaks,
            "results": [], "offset_votes": {}, "y": y,
        }

    top_count, top_sid, top_offset = results[0]
    return {
        "prediction": song_names[top_sid],
        "votes": top_count,
        "S_db": S_db,
        "peaks": peaks,
        "results": [(v, song_names[sid], off) for v, sid, off in results],
        "winning_offsets": offset_votes.get(top_sid, Counter()),
        "y": y,
    }
