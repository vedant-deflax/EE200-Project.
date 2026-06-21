"""
Q3B — Zapptain America
A small Streamlit front-end around the fingerprinting logic in fingerprint.py.

Two modes (pick from the sidebar):
  1. Single clip  — upload one query, see the spectrogram / constellation /
                     offset-histogram and the predicted song.
  2. Batch        — upload several queries at once, get a downloadable
                     results.csv with exactly two columns: filename, prediction.

The song database (database.pkl) is built ahead of time with build_database.py
and shipped alongside this file, so the app doesn't need to re-index the mp3s
every time it starts up.
"""
import io
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import streamlit as st

from fingerprint import (
    load_database, load_audio, fingerprint_signal, match_query, identify, SR, N_FFT, HOP,
)

st.set_page_config(page_title="Song ID — Q3B", layout="wide")


@st.cache_resource
def get_database():
    return load_database("database.pkl")


database, song_names = get_database()


def plot_spectrogram_and_constellation(S_db, peaks):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    librosa.display.specshow(S_db, sr=SR, hop_length=HOP, x_axis="time", y_axis="hz", ax=ax, cmap="gray_r")
    t_axis = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=SR, hop_length=HOP)
    f_axis = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    pt = [t_axis[t] for t, f in peaks]
    pf = [f_axis[f] for t, f in peaks]
    ax.scatter(pt, pf, s=8, c="red")
    ax.set_ylim(0, 4000)
    ax.set_title("Spectrogram with constellation peaks")
    fig.tight_layout()
    return fig


def plot_offset_histogram(winning_offsets, song_name):
    fig, ax = plt.subplots(figsize=(6, 3))
    if winning_offsets:
        offsets = list(winning_offsets.elements())
        ax.hist(offsets, bins=40)
        best_offset = winning_offsets.most_common(1)[0][0]
        ax.axvline(best_offset, color="red", linestyle="--", label=f"offset = {best_offset}")
        ax.legend()
    ax.set_xlabel("offset (frames)")
    ax.set_ylabel("matching hash count")
    ax.set_title(f"Offset histogram — {song_name}")
    fig.tight_layout()
    return fig


st.title("🎵 Song Identifier")
st.caption(f"Database indexed from {len(song_names)} songs · {sum(len(v) for v in database.values())} hash entries")

mode = st.sidebar.radio("Mode", ["Single clip", "Batch"])

if mode == "Single clip":
    st.subheader("Single-clip identification")
    query_file = st.file_uploader("Upload a short query clip (wav / mp3)", type=["wav", "mp3", "m4a"])

    if query_file is not None:
        with st.spinner("fingerprinting..."):
            buf = io.BytesIO(query_file.read())
            result = identify(buf, database, song_names)

        if result["prediction"] is None:
            st.error("No matching hashes found at all — clip may be too short or too noisy.")
        else:
            st.success(f"**Predicted song:** {result['prediction']}  (votes: {result['votes']})")

            st.markdown("**Top candidates**")
            cand_df = pd.DataFrame(result["results"], columns=["votes", "song", "offset_frames"])
            st.dataframe(cand_df, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_spectrogram_and_constellation(result["S_db"], result["peaks"]))
            with col2:
                st.pyplot(plot_offset_histogram(result["winning_offsets"], result["prediction"]))

elif mode == "Batch":
    st.subheader("Batch identification")
    st.write("Upload several query clips. The app fingerprints each one, writes "
             "`results.csv` with `filename, prediction` columns, and lets you download it.")

    query_files = st.file_uploader(
        "Upload query clips", type=["wav", "mp3", "m4a"], accept_multiple_files=True
    )

    if query_files:
        rows = []
        progress = st.progress(0.0)
        for i, qf in enumerate(query_files):
            buf = io.BytesIO(qf.read())
            try:
                y = load_audio(buf)
                _, peaks, _ = fingerprint_signal(y)
                results, _ = match_query(peaks, database)
                prediction = song_names[results[0][1]] if results else ""
            except Exception as e:
                prediction = ""
                st.warning(f"could not process {qf.name}: {e}")

            filename_no_ext = os.path.splitext(qf.name)[0]
            rows.append({"filename": filename_no_ext, "prediction": prediction})
            progress.progress((i + 1) / len(query_files))

        results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(results_df, use_container_width=True)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results.csv", data=csv_bytes, file_name="results.csv", mime="text/csv")
