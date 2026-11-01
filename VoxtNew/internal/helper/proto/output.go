package proto

// ---- output.* ----
//
// The system's audio output, muted while a recording runs (Settings →
// Audio → Mute system audio while recording). Only the mute property of
// the default output device changes; nothing of the system's sound is
// read, so no permission is needed. Go decides when (after the start cue,
// until the recording stops) and what to do when the device changes or the
// user unmutes; the helper reads and writes Core Audio, and undoes the
// mutes it made (its claims) when the app goes away: stdin closing,
// helper.shutdown, a termination signal, the Mac going to sleep. A claim
// is written to a journal file before the mute, so that a helper that
// crashed leaves it for the next one (output.recover).

// OutputDevice is an output device. UID is its identity: the helper finds
// the device by it, as its ID may change when the device is reconnected
// or Core Audio restarts, and an ID may be reused by another device.
type OutputDevice struct {
	ID   uint32 `json:"id"`
	UID  string `json:"uid"`
	Name string `json:"name,omitempty"`
}

// OutputState is the default output device and its mute: Muted is absent
// when the device has no mute it can read (or is gone); Settable says
// whether the helper can change it.
type OutputState struct {
	Device   *OutputDevice `json:"device,omitempty"` // absent: no default output
	Muted    *bool         `json:"muted,omitempty"`
	Settable bool          `json:"settable"`
}

// IsMuted reports whether the state says muted (false when unknown).
func (s OutputState) IsMuted() bool { return s.Muted != nil && *s.Muted }

// OutputWatchParams turns the watch on or off: while on, the helper sends
// output.changed when the default output device or its mute changes. The
// result is the OutputState now.
type OutputWatchParams struct {
	On bool `json:"on"`
}

// OutputMuteParams mutes Device and takes a claim on it, when it is not
// muted already: a mute the user made is never claimed. The claim is
// written to a file in Journal (a directory; "" keeps no journal) before
// the mute. Errors: unsupported (no mute it can set), device_unavailable
// (no device has that UID now), internal (the journal could not be
// written: nothing was muted).
type OutputMuteParams struct {
	Device  OutputDevice `json:"device"`
	Journal string       `json:"journal,omitempty"`
}

// OutputMuteResult says whether the helper muted the device and claimed
// it (false: it was muted already), and its state after.
type OutputMuteResult struct {
	Took  bool        `json:"took"`
	State OutputState `json:"state"`
}

// OutputReleaseParams gives the claim on Device back; with Unmute the
// helper unmutes it if it is still muted. Without a claim on it the
// helper does nothing: it undoes only its own mutes, so the app may
// release a claim it is not sure was taken (a timed-out output.mute).
type OutputReleaseParams struct {
	Device OutputDevice `json:"device"`
	Unmute bool         `json:"unmute,omitempty"`
}

// OutputReleaseResult says whether the helper unmuted the device.
type OutputReleaseResult struct {
	Unmuted bool `json:"unmuted"`
}

// OutputRecoverParams finds the claims helpers that are gone left in
// Journal (they crashed or were killed) and undoes them: each device
// still muted is unmuted, and the files are removed. The app sends it
// after each helper.hello, before any output.mute.
type OutputRecoverParams struct {
	Journal string `json:"journal"`
}

// OutputRecoverResult lists the devices unmuted.
type OutputRecoverResult struct {
	Restored []OutputDevice `json:"restored,omitempty"`
}

// OutputReleasedParams (notification) says the helper gave its claims
// back on its own, and why: OutputReasonSleep, before the Mac sleeps.
// Devices are the claims it gave back (each unmuted if it was still
// muted).
type OutputReleasedParams struct {
	Devices []OutputDevice `json:"devices,omitempty"`
	Reason  string         `json:"reason"`
}

// OutputReasonSleep: NSWorkspace's willSleep.
const OutputReasonSleep = "sleep"
