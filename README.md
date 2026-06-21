# Q3B — Song Identifier App

Streamlit wrapper around the fingerprinting pipeline from Q3A (spectrogram ->
constellation peaks -> paired hashes -> offset-histogram matching).

## Files
- `fingerprint.py`    — core algorithm (shared with the Q3A notebook's logic)
- `build_database.py` — one-off script: indexes the mp3 folder into `database.pkl`
- `database.pkl`      — pre-built index of the 50 provided songs (ships with the app
                          so it works immediately, no re-indexing on startup)
- `app.py`            — the Streamlit app itself (single-clip mode + batch mode)
- `requirements.txt`

## Running locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the URL it prints (usually http://localhost:8501).

If you ever need to rebuild the database (e.g. you add/replace songs):
```bash
python build_database.py "/path/to/EE200 Project Song Database"
```
This overwrites `database.pkl`. Re-run before redeploying.

## Deploying on Streamlit Community Cloud
1. Push this folder (including `database.pkl`) to a public GitHub repo.
   `database.pkl` is ~24 MB, well within GitHub's normal file size limit, so no
   Git LFS needed.
2. Go to https://share.streamlit.io, sign in with GitHub, "New app", point it at
   this repo and `app.py` as the entry point.
3. Streamlit Cloud installs `requirements.txt` automatically. First boot may take
   a minute or two while librosa's dependencies install, after that it's quick
   because `database.pkl` is loaded directly (no re-indexing).
4. Test both modes on the live URL before submitting:
   - Single clip: upload a short mp3/wav cut from one of the songs, confirm the
     spectrogram / constellation / offset histogram all render and the right
     song is predicted.
   - Batch: upload a handful of query clips at once, confirm `results.csv`
     downloads with the right `filename, prediction` columns.

## Note on librosa/ffmpeg on Streamlit Cloud
librosa needs `ffmpeg` to decode mp3 via audioread as a fallback. Streamlit
Cloud's default image usually has it, but if mp3 uploads fail in the deployed
app and not locally, add a `packages.txt` file (one line: `ffmpeg`) to the repo
root — Streamlit Cloud installs that via apt before your requirements.
