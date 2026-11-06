package dsp

import "math"

// DefaultRate is the rate used when a caller passes rate <= 0.
const DefaultRate = 16000

// DefaultHighPassHz is the high-pass filter's corner.
const DefaultHighPassHz = 80

const minDB = -120.0

// Params tune the processing; a zero field takes its default (Defaults()).
// Levels are dBFS, gains dB.
type Params struct {
	// HighPassOff and LevelOff turn one stage off (troubleshooting; the
	// settings have one switch for both).
	HighPassOff, LevelOff bool
	HighPassHz            float64 // the high-pass corner: 80
	// TargetDB is the speech level (the RMS of its frames) a quiet voice
	// is raised to, by at most MaxGainDB: -23 and 12. Speech louder than
	// LoudDB is lowered to it, by at most MaxCutDB: -12 and 6. Speech in
	// between is left as it is.
	TargetDB  float64
	MaxGainDB float64
	LoudDB    float64
	MaxCutDB  float64
	// NoiseCeilDB is the highest a gain may lift the noise floor: -50.
	NoiseCeilDB float64
	// CeilingDB is the limiter's: no sample goes out louder (-1). LimitDB
	// is how much the limiter may take off the speech's peaks (their 99th
	// percentile): a gain that would need more is smaller (6).
	CeilingDB float64
	LimitDB   float64
	// Streaming: the gain's time constants going up and down (500, 100
	// ms), the time the speech level is averaged over after its first
	// second (3000 ms); the limiter's release (50 ms).
	RiseMS, FallMS, TrackMS, ReleaseMS int
}

// Defaults are the tuned parameters.
func Defaults() Params {
	return Params{
		HighPassHz:  DefaultHighPassHz,
		TargetDB:    -23,
		MaxGainDB:   12,
		LoudDB:      -12,
		MaxCutDB:    6,
		NoiseCeilDB: -50,
		CeilingDB:   -1,
		LimitDB:     6,
		RiseMS:      500,
		FallMS:      100,
		TrackMS:     3000,
		ReleaseMS:   50,
	}
}

// Resolved fills zero (and out of range) fields with their defaults:
// levels must be below 0 dBFS, gains and times above 0; LoudDB is at least
// TargetDB.
func (p Params) Resolved() Params {
	d := Defaults()
	pos := func(v *float64, def, hi float64) {
		if !(*v > 0) || *v > hi { // also NaN
			*v = def
		}
	}
	neg := func(v *float64, def float64) {
		if !(*v < 0) || *v < minDB {
			*v = def
		}
	}
	ms := func(v *int, def int) {
		if *v <= 0 {
			*v = def
		}
	}
	pos(&p.HighPassHz, d.HighPassHz, 1000)
	neg(&p.TargetDB, d.TargetDB)
	pos(&p.MaxGainDB, d.MaxGainDB, 40)
	neg(&p.LoudDB, d.LoudDB)
	p.LoudDB = max(p.LoudDB, p.TargetDB)
	pos(&p.MaxCutDB, d.MaxCutDB, 40)
	neg(&p.NoiseCeilDB, d.NoiseCeilDB)
	neg(&p.CeilingDB, d.CeilingDB)
	pos(&p.LimitDB, d.LimitDB, 40)
	ms(&p.RiseMS, d.RiseMS)
	ms(&p.FallMS, d.FallMS)
	ms(&p.TrackMS, d.TrackMS)
	ms(&p.ReleaseMS, d.ReleaseMS)
	return p
}

// minGainDB is the smallest gain worth applying: under it the speech is
// left as it is.
const minGainDB = 0.5

// GainFor is the gain for speech at speechDB over a noise floor at floorDB,
// with its peaks at peakDB (p resolved): up to TargetDB when quieter, by
// at most MaxGainDB, without lifting the floor over NoiseCeilDB or asking
// the limiter for more than LimitDB; down to LoudDB when louder, by at
// most MaxCutDB; else 0. No speech (speechDB <= -120): 0.
func (p Params) GainFor(speechDB, floorDB, peakDB float64) float64 {
	var g float64
	switch {
	case speechDB <= minDB || math.IsNaN(speechDB):
		return 0
	case speechDB < p.TargetDB:
		g = min(p.TargetDB-speechDB, p.MaxGainDB, p.NoiseCeilDB-floorDB, p.CeilingDB+p.LimitDB-peakDB)
		g = max(g, 0)
	case speechDB > p.LoudDB:
		g = max(p.LoudDB-speechDB, -p.MaxCutDB)
	}
	if math.Abs(g) < minGainDB {
		return 0
	}
	return g
}

// Report says what the processing did.
type Report struct {
	HighPass bool    // the high-pass filter ran
	Level    bool    // the level stage ran
	GainDB   float64 // the gain on the speech (a stream: at its end)
	// MinGainDB and MaxGainDB are the extremes a stream's gain took after
	// it heard speech (a file: GainDB).
	MinGainDB, MaxGainDB float64
	// SpeechDB is the speech's level before the gain, PeakDB its peaks',
	// FloorDB the noise floor (after the high-pass); -120 when no speech
	// was heard.
	SpeechDB, PeakDB, FloorDB float64
	// Limited counts the samples the limiter turned down, Clipped those
	// the high-pass clamped.
	Limited, Clipped int
}

func dbToGain(db float64) float64 { return math.Pow(10, db/20) }
