package main

import "encoding/binary"

// pcm16Mono is 16-bit PCM at rate with channels as 16-bit mono at target
// Hz: the channels averaged, the rate linearly resampled.
func pcm16Mono(target int, pcm []byte, rate, channels int) []byte {
	if channels <= 0 {
		channels = 1
	}
	if rate <= 0 {
		rate = target
	}
	frames := len(pcm) / (2 * channels)
	if frames == 0 {
		return nil
	}
	if frames == 1 {
		out := make([]byte, 2)
		copy(out, pcm[:2])
		return out
	}
	mono := make([]int16, frames)
	for i := range mono {
		sum := 0
		for ch := 0; ch < channels; ch++ {
			sum += int(int16(binary.LittleEndian.Uint16(pcm[(i*channels+ch)*2:])))
		}
		mono[i] = int16(sum / channels)
	}
	if rate == target {
		out := make([]byte, len(mono)*2)
		for i, s := range mono {
			binary.LittleEndian.PutUint16(out[i*2:], uint16(s))
		}
		return out
	}
	// Linear resample rate → target.
	n := int(int64(frames) * int64(target) / int64(rate))
	out := make([]byte, n*2)
	step := float64(rate) / float64(target)
	for i := 0; i < n; i++ {
		x := float64(i) * step
		j := int(x)
		if j+1 >= frames {
			j = frames - 2
			x = float64(j)
		}
		if j < 0 {
			j = 0
		}
		frac := x - float64(j)
		v := float64(mono[j])*(1-frac) + float64(mono[j+1])*frac
		binary.LittleEndian.PutUint16(out[i*2:], uint16(int16(v)))
	}
	return out
}
