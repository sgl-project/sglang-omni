// Package vad is a small energy voice activity detector for VoxtNew's
// recordings (PCM16 mono, little-endian, any rate; 16 kHz in practice).
// It is pure DSP with no model, and it is deliberately conservative: when it
// is unsure it keeps audio. Sending a second of silence costs a little
// time; cutting the first syllable of a word costs a wrong transcript.
//
// # Detector
//
// The audio is cut into frames of FrameMS (20 ms). A frame's level is
// 10·log10(mean(x²)) dBFS with x in [-1, 1], floored at -120 dB.
//
// A noise floor follows the quiet frames: it falls quickly (half-way to the
// frame each frame) when a frame is below it and rises slowly (a time
// constant of 0.75 s) when a frame that is not speech is above it. It never
// rises during speech, so a noise that grows suddenly by more than OnsetDB
// is taken for speech and kept, which is the safe way to be wrong. Frames
// of digital silence (below -96 dBFS, under one bit of dither) do not move
// it: a microphone that starts with a few buffers of zeros must not pull the
// floor down to -120 dB and make the room's noise look like speech.
//
// Speech starts when every frame for OnsetMS (60 ms, which rejects key
// clicks) is above floor+OnsetDB and above MinLevelDB; the segment starts
// at the first of those frames. It goes on while frames stay at or above
// floor+OffsetDB (hysteresis: the threshold to stay is lower than the one
// to start) and ends after HangoverMS (300 ms) below it; the segment ends
// at the end of the last frame at or above the threshold.
//
// The Detector is causal and starts from InitFloorDB (-50 dBFS). Analyze,
// which sees the whole recording, starts the floor at the 10th percentile
// of the frame levels instead (never above InitFloorDB), so a recording
// that starts with speech is not missing its beginning.
//
// # Compact (file ASR)
//
// Compact keeps [first speech − LeadMS, last speech + TailMS] and shortens
// every pause between segments longer than LongGapMS (1 s) to GapKeepMS
// (400 ms: half after the speech before it, half before the next). Shorter
// pauses are kept as they are: they carry the rhythm and punctuation the
// recognisers rely on. Each cut gets a 4 ms linear fade so the joins do
// not click. Unlike Voxt, which drops every frame its VAD marks silent, it
// never removes audio inside or close to speech. When compacting would save
// less than MinSavingMS, Compact returns the original audio unchanged.
// When there is no speech it returns the original too, with Changed false:
// the caller looks at Analysis.Speech and does not send a request.
//
// # Gate (streaming ASR)
//
// A Gate holds a stream's leading silence: nothing goes out until speech
// starts; then PreRollMS of what came before the onset and everything
// after, pauses included (a live stream needs a continuous timeline). It
// opens anyway after MaxHoldMS of audio without speech, from PreRollMS
// before that point, so a stream is never held back for long.
//
// # Two versions of a recording
//
// When the speech model gets a filtered recording (internal/dsp), the
// detection runs on both the filtered and the original audio and keeps the
// speech either finds: Merge joins two analyses' segments, and Gate.Hear
// lets a gate listen to the original while it holds and sends the filtered
// stream. The filter takes hum out of the noise floor, so the filtered
// audio shows quiet speech the original hides; the original keeps what a
// strong hum made all speech. Either way a filter only makes it keep more.
package vad
