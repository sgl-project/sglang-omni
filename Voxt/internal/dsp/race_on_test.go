//go:build race

package dsp

// raceOn: the race detector slows the processing about ninefold, so
// timings under it say nothing about the code's speed.
const raceOn = true
