package proto

// ---- speech.* ----

// SpeechParams starts a speech.recognize job: macOS's own speech
// recognition (SFSpeechRecognizer), on this Mac only, of the audio that
// speech.audio brings. The job sends speech.started once the recognizer
// runs, then speech.partial while it hears, and answers with a
// SpeechResult once the audio up to speech.audio's End is recognised;
// job.cancel discards it. It fails (and sends no speech.started) with
// permission_denied when Speech Recognition is not allowed (not asked
// yet: macOS asks, and the job waits for the answer) and unsupported
// when macOS has no on-device model for the language.
type SpeechParams struct {
	// Lang is the language spoken, a BCP-47 tag ("zh", "en-GB",
	// "zh-Hant"); the helper picks the recognizer's locale for it. "":
	// the Mac's own dictation language.
	Lang string `json:"lang,omitempty"`
	// SampleRate of speech.audio's PCM (default 16000).
	SampleRate int `json:"sampleRate,omitempty"`
	// Hints are words and phrases to favour (the dictionary's terms).
	Hints []string `json:"hints,omitempty"`
}

// SpeechStartedParams: the job's recognizer runs, for Locale ("zh-CN").
type SpeechStartedParams struct {
	Job    string `json:"job"`
	Locale string `json:"locale"`
}

// SpeechPartialParams is the text heard so far: the whole hypothesis,
// not a delta.
type SpeechPartialParams struct {
	Job  string `json:"job"`
	Text string `json:"text"`
}

// SpeechResult is the final text; "" when nothing was said.
type SpeechResult struct {
	Text string `json:"text"`
}

// SpeechAudioParams brings a speech.recognize job audio: mono PCM16
// little endian at its SampleRate (base64 in JSON), in order. End: no
// more audio comes; the job answers once it is recognised. A job that
// is not running (it ended, or failed) is bad_request.
type SpeechAudioParams struct {
	Job  string `json:"job"`
	Data []byte `json:"data,omitempty"`
	End  bool   `json:"end,omitempty"`
}
