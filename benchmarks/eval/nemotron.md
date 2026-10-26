# Nemotron ASR benchmark

A unified measurement method for Nemotron 3.5 ASR over HTTP and native streaming.
Run the commands from the repository root with the project and benchmark
dependencies installed. Native streaming requires `websockets>=14`.

## Prepare audio

Convert SeedTTS recordings to mono 16 kHz PCM16 for both endpoints:

```bash
python -m benchmarks.eval.prepare_nemotron_audio \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --language en --max-samples 20 \
  --output benchmarks/results/nemotron-input
```

Omit `--max-samples` for the full split. Use `--language zh` for Chinese.
For local audio, pass `--meta /path/to/meta.lst`. Each line has the format
`sample_id|reference_text|audio_path|target_text`, with unique sample IDs and
audio paths relative to the metadata file.

## Start the service

```bash
MODEL_PATH=$(hf download nvidia/nemotron-3.5-asr-streaming-0.6b \
  --revision ea30d66debe3740a08b573244286791d423d6b3e --quiet)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
sgl-omni serve --model-path "$MODEL_PATH" \
  --model-name nvidia/nemotron-3.5-asr-streaming-0.6b \
  --enable-realtime --host 127.0.0.1 --port 8000 \
  --asr.factory.dtype float32 --asr.factory.num_lookahead_tokens 3 \
  --asr.factory.max_batch_size 8 --asr.factory.max_batch_wait_ms 2 \
  --asr.factory.session_max_concurrency 8
```

## Measure HTTP and native streaming

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --meta benchmarks/results/nemotron-input/meta.lst \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b --port 8000 \
  --request-language auto --lang en \
  --concurrencies 1,4 --repeats 3 --warmup \
  --save-raw-dir benchmarks/results/nemotron-http-raw \
  --output benchmarks/results/nemotron-http.json

python -m benchmarks.eval.benchmark_asr_realtime \
  --meta benchmarks/results/nemotron-input/meta.lst \
  --protocol native --port 8000 --lang en --max-samples 20 \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --concurrencies 1,4 --repeats 3 --warmup \
  --packet-ms 20 --timeout-s 300 \
  --output benchmarks/results/nemotron-native-paced.json
```

Set native `--max-samples` to the prepared sample count. For Chinese, set
`--lang zh` in both commands. HTTP `--request-language auto` enables automatic
language detection while `--lang` selects the language used for scoring.

The example uses 20 ms audio packets. The shared realtime CLI defaults to
200 ms. Keep packet size and pacing consistent when comparing runs.
Native streaming sends audio at recording speed by default. Add `--no-pace`
to measure throughput with audio sent as quickly as possible.

`--concurrencies` controls simultaneous requests. `--repeats` sets the measured
runs per concurrency. `--warmup` adds an unmeasured run before those repeats.

## Results

HTTP saves aggregate results to `--output` and per-sample results to
`--save-raw-dir`. Native streaming saves results to `--output` and audio,
received events and timestamps to `OUTPUT.traces`. Replay a saved trace with:

```bash
python -m benchmarks.realtime_asr.replay path/to/trace.json
```

| Metric | Meaning |
|---|---|
| WER / CER | Word error rate for English or character error rate for Chinese. `corpus_wer` is a fraction. |
| Success / evaluated | Completed requests and requests with a usable reference for scoring. |
| Audio seconds per second (RTFx) | Successfully processed audio duration divided by total run time. |
| Requests per second | Successful requests divided by total run time. |
| First text (`first_text_s`) | Seconds from the first audio packet to the first nonempty partial text. Null if no partial text arrives. |
| EOS to final / drained (`final_latency_s` / `done_to_completed_s`) | Seconds from end-of-input to final text or confirmation that all input was processed. |
| Request wall time / RTF | Native request duration from first audio send to input drained, and that duration divided by audio duration. |

Latency summaries include successful requests only. Failed requests remain in
the saved results. Check per-sample errors and scoring coverage in HTTP results,
as the HTTP command can exit successfully even when requests fail.

With `--no-pace`, EOS-to-final includes queued audio processing time.
HTTP file transcription and native streaming use different inference paths,
so compare results within the same mode and configuration.
