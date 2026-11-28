"""Extract the SeedTTS English split into reference WAVs plus a manifest.

The Arrow parquet embeds each prompt clip; the server wants a server-local
path, so this writes one WAV per sample and records the fixed evaluation order.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib

import pyarrow.parquet as pq
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(
        args.parquet,
        columns=["sample_id", "ref_text", "ref_audio_path", "target_text", "ref_audio"],
    )
    rows = []
    for index in range(table.num_rows):
        sample_id = table["sample_id"][index].as_py()
        audio = table["ref_audio"][index].as_py()
        wav_bytes = audio["bytes"]
        wav_path = out_dir / f"{sample_id}.wav"
        if not wav_path.exists():
            wav_path.write_bytes(wav_bytes)
        info = sf.info(io.BytesIO(wav_bytes))
        rows.append(
            {
                "index": index,
                "sample_id": sample_id,
                "target_text": table["target_text"][index].as_py(),
                "ref_text": table["ref_text"][index].as_py(),
                "ref_wav": str(wav_path),
                "ref_dur_s": info.duration,
                "ref_sr": info.samplerate,
            }
        )

    pathlib.Path(args.manifest).write_text(json.dumps(rows))
    total_ref = sum(row["ref_dur_s"] for row in rows)
    print(
        f"wrote {len(rows)} refs, {total_ref:.1f}s reference audio -> {args.manifest}"
    )


if __name__ == "__main__":
    main()
