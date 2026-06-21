"""
Run this once (locally, before deploying) to index the whole song folder
and produce database.pkl. The Streamlit app just loads that pickle at
startup instead of re-fingerprinting 50 mp3s on every reboot.

Usage:
    python build_database.py /path/to/EE200\ Project\ Song\ Database
"""
import sys
import glob
import os
import time

from fingerprint import build_database, save_database

def main():
    song_dir = sys.argv[1] if len(sys.argv) > 1 else "songs"
    song_files = sorted(glob.glob(os.path.join(song_dir, "*.mp3")))
    if not song_files:
        print(f"no mp3 files found in {song_dir}")
        sys.exit(1)

    print(f"indexing {len(song_files)} songs from {song_dir} ...")
    t0 = time.time()
    database, song_names = build_database(song_files)
    print(f"done in {time.time()-t0:.1f}s — {len(database)} unique hashes")

    save_database(database, song_names, "database.pkl")
    size_mb = os.path.getsize("database.pkl") / 1e6
    print(f"saved database.pkl ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
