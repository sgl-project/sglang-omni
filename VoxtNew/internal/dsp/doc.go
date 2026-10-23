// Package dsp is the light processing of what a speech model gets of a
// recording (PCM16 mono, little-endian; 16 kHz in practice): pure Go, no
// model and no cgo. The recording itself is never changed: every function
// returns a new slice, and the history keeps the original.
//
// # The chain
//
// capture → high-pass → voice activity detection → level → (a file) trim,
// or (a stream) gate → the speech model. The detection runs after the filter
// so that hum does not raise its noise floor, and on the original as
// well: what either finds is speech (vad.Merge, vad.Gate.Hear), so the
// filter can only make it keep more. The level comes after the detection
// so that its gain does not move the detection's thresholds; in a stream
// it comes after the gate, whose opening chunk (the pre-roll and the
// onset) gives it its first gain.
//
// # High-pass
//
// A fourth-order Butterworth high-pass (two RBJ biquads in transposed
// direct form II, float64) at 80 Hz: -16 dB at 50 Hz, -10 dB at 60 Hz,
// -3 dB at 80 Hz, -0.7 dB at 100 Hz, nothing from 200 Hz on; no DC. It
// keeps its state, so chunks come out as the whole would, and its first
// call settles it on the mean of the first 20 ms: an offset does not ring
// at the start.
//
// # Level
//
// The speech's level is the power mean of the frames the detection put in
// speech (at or over the floor + OffsetDB). Params.GainFor turns it into a
// gain: speech under TargetDB (-23 dBFS) is raised to it, by at most
// MaxGainDB (12), never lifting the noise floor over NoiseCeilDB (-50) or
// asking the limiter to take more than LimitDB (6) off the speech's peaks;
// speech over LoudDB (-12) is lowered to it, by at most MaxCutDB (6);
// between, and under 0.5 dB, nothing changes. A file gets one gain
// (ProcessFile); a stream's Leveler measures each chunk before it turns
// it, holds the gain through silence (it measures speech only, so it
// never creeps up on the noise), sets it at once on the first speech and
// then moves it with RiseMS (500 ms) and FallMS (100 ms), toward the level
// of the first second of speech, then of the last TrackMS (3 s) of it.
//
// # Limiter
//
// Last, a peak limiter without look-ahead: a sample that would pass
// CeilingDB (-1 dBFS) sets the gain that puts it there, which recovers
// over ReleaseMS (50 ms). Nothing goes out over the ceiling, whatever the
// gain and the input (the filter clamps only an input at full scale with
// strong low frequencies, and counts it).
package dsp
