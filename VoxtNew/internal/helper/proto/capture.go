package proto

// ---- capture.* ----

// DevicesResult answers capture.devices.
type DevicesResult struct {
	Devices []CaptureDevice `json:"devices"`
}

// CaptureDevice is an audio input.
type CaptureDevice struct {
	ID         string `json:"id"` // Core Audio device UID, stable across restarts
	Name       string `json:"name"`
	Channels   int    `json:"channels"`
	SampleRate int    `json:"sampleRate"`
	Default    bool   `json:"default"`             // the system's default input
	Transport  string `json:"transport,omitempty"` // "builtin" | "usb" | "bluetooth" | "continuity" | "aggregate" | "virtual"
}

// CaptureParams starts recording. The capture.start job sends
// capture.started once the audio engine runs, then capture.level and
// capture.pcm, and answers with a CaptureResult when capture.stop,
// MaxSeconds or the input ends it; job.cancel discards it. It fails (and
// sends no capture.started) when no input starts: permission_denied,
// device_unavailable, busy.
type CaptureParams struct {
	// Devices are device ids in the user's order of preference: the first
	// one present that starts records, else the system's default input
	// (CaptureStarted.FellBack). Empty: the default input.
	Devices []string `json:"devices,omitempty"`
	// SampleRate of capture.pcm (default 16000). The audio is always mono
	// PCM16 little endian.
	SampleRate int `json:"sampleRate,omitempty"`
	// FrameMS is the audio in each capture.pcm (default 100); LevelHz the
	// rate of capture.level (default 20).
	FrameMS int `json:"frameMs,omitempty"`
	LevelHz int `json:"levelHz,omitempty"`
	// MaxSeconds ends the recording on its own (0 = until stopped): a
	// guard against a capture.stop that never comes.
	MaxSeconds float64 `json:"maxSeconds,omitempty"`
}

// CaptureStartedParams: the job records. Until it comes the user's first
// words are not being heard (200–400 ms after capture.start).
type CaptureStartedParams struct {
	Job        string `json:"job"`
	Device     string `json:"device"` // the device that records
	DeviceName string `json:"deviceName"`
	// FellBack: not the first choice, which failed (FallbackReason).
	FellBack         bool    `json:"fellBack"`
	FallbackReason   string  `json:"fallbackReason,omitempty"`
	DeviceSampleRate int     `json:"deviceSampleRate"`
	DeviceChannels   int     `json:"deviceChannels"`
	SampleRate       int     `json:"sampleRate"` // of capture.pcm
	StartMS          float64 `json:"startMs"`    // capture.start to the engine running
}

// LevelParams is the input meter, in dBFS (-120 is silence).
type LevelParams struct {
	Job     string  `json:"job"`
	Seconds float64 `json:"seconds"` // of audio since the start
	RMS     float64 `json:"rms"`
	Peak    float64 `json:"peak"`
}

// PCMParams carries recorded audio: mono PCM16 little endian at the
// started sample rate. Seq counts from 0 without gaps; Offset is the
// sample offset of its first sample. Data is base64 in JSON (Go []byte,
// Swift Data). The last frame may be shorter.
type PCMParams struct {
	Job    string `json:"job"`
	Seq    int    `json:"seq"`
	Offset int    `json:"offset"`
	Data   []byte `json:"data"`
}

// RestartedParams: the job's audio engine was rebuilt (Reason
// CaptureNoCallback, CaptureConfigurationChange or CaptureDeviceLost for
// the default input, which follows the new default). Recording goes on.
type RestartedParams struct {
	Job     string `json:"job"`
	Reason  string `json:"reason"`
	Attempt int    `json:"attempt"`
	Device  string `json:"device"` // the device it records from now
}

// CaptureResult describes the recording. Every capture.pcm of the job
// was sent before it.
type CaptureResult struct {
	Device string `json:"device"` // the last device that recorded
	// Reason is why it ended: CaptureStopped, CaptureMaxSeconds,
	// CaptureDeviceLost (the chosen input went away) or CaptureNoCallback
	// (the input stopped delivering audio, restarts did not help).
	Reason   string  `json:"reason"`
	Frames   int     `json:"frames"`  // capture.pcm notifications sent
	Samples  int     `json:"samples"` // at the started sample rate
	Seconds  float64 `json:"seconds"`
	Restarts int     `json:"restarts"`
	Peak     float64 `json:"peak"` // dBFS
	// FirstFrameMS is capture.start to the first capture.pcm (0: none).
	FirstFrameMS float64 `json:"firstFrameMs,omitempty"`
	CPUSeconds   float64 `json:"cpuSeconds"` // the helper's CPU time while it recorded
}

// Capture end and restart reasons.
const (
	CaptureStopped             = "stopped"
	CaptureMaxSeconds          = "maxSeconds"
	CaptureDeviceLost          = "deviceLost"
	CaptureNoCallback          = "noCallback"
	CaptureConfigurationChange = "configurationChange"
)

// CaptureStopParams names the capture.start job to stop.
type CaptureStopParams struct {
	Job string `json:"job"`
}

// CaptureStopResult says whether the capture was running; its job then
// answered (before this response).
type CaptureStopResult struct {
	Stopped bool `json:"stopped"`
}
