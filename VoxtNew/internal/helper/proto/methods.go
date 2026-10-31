package proto

// Methods. The shapes follow Auralink's spikes on a Mac (its
// research/spikes.md): the helper records, watches the keyboard, reads and
// writes text in the app in front, reads browser URLs and mutes the
// system's output while recording; Go keeps every policy and passes it in
// the params.
//
// Requests marked "job" are long-running: job.cancel stops them, and
// their notifications name them in "job". Other requests are control
// requests, answered promptly, even while a job runs.
//
// Field conventions, which the fixtures hold both sides to: a Go field
// without omitempty is always on the wire (Swift: a non-optional); a Go
// field with omitempty is absent when zero (Swift: an optional left nil).
const (
	MethodHello     = "helper.hello"    // request; must be first
	MethodShutdown  = "helper.shutdown" // request: cancel jobs, answer, exit 0
	MethodJobCancel = "job.cancel"      // request

	MethodPermissionsStatus  = "permissions.status"  // request: the Microphone and Accessibility, as the helper sees them
	MethodPermissionsRequest = "permissions.request" // request: show macOS's prompt for one, then report

	MethodCaptureDevices = "capture.devices" // request: the audio inputs
	MethodCaptureStart   = "capture.start"   // request, job: records until capture.stop
	MethodCaptureStop    = "capture.stop"    // request: ends a capture.start, which answers first

	MethodHotkeyConfigure = "hotkey.configure" // request: the bindings the event tap watches (replaces all)
	MethodHotkeyStatus    = "hotkey.status"    // request: the tap's state and counters

	MethodAppFrontmost = "app.frontmost" // request: the app in front

	MethodTextFocus     = "textio.focus"     // request: the focused element of the app in front (can it take text?)
	MethodTextSelection = "textio.selection" // request: the selected text (Accessibility, else ⌘C when allowed)
	MethodTextPaste     = "textio.paste"     // request: insert text through the clipboard and ⌘V, then restore it

	MethodAppList      = "app.list"      // request: running (and installed) apps, to pick from
	MethodAppConfigure = "app.configure" // request: which apps get AXManualAccessibility as they come to the front
	MethodAppIcons     = "app.icons"     // request: apps' icons as PNG, for the lists that name apps

	MethodBrowserURL = "browser.url" // request: the front tab's URL of a browser (Accessibility only)

	MethodOutputWatch   = "output.watch"   // request: the default output and its mute; output.changed while on
	MethodOutputMute    = "output.mute"    // request: mute a device and claim it (not when it was muted already)
	MethodOutputRelease = "output.release" // request: give a claim back (unmuting it, unless the user did)
	MethodOutputRecover = "output.recover" // request: undo the claims a helper that is gone left in the journal

	// Notifications, helper → app.
	MethodCaptureStarted   = "capture.started"   // a capture.start job records: which device, how long it took
	MethodCaptureLevel     = "capture.level"     // input meter of a capture.start job, ~20 Hz
	MethodCapturePCM       = "capture.pcm"       // audio of a capture.start job, a frame every ~100 ms
	MethodCaptureRestarted = "capture.restarted" // the job's audio engine was rebuilt (watchdog, device change)
	MethodHotkeyEvent      = "hotkey.event"      // a binding fired (or Escape while active)
	MethodHotkeyState      = "hotkey.state"      // the event tap was installed or lost (e.g. Accessibility granted)
	MethodAppActivated     = "app.activated"     // another app came to the front
	MethodOutputChanged    = "output.changed"    // the default output device or its mute changed (output.watch on)
	MethodOutputReleased   = "output.released"   // the helper gave its claims back on its own (sleep)
	MethodLog              = "log"               // a log line (stderr stays free-form text)
)

// HelloParams opens a session. The app sends it first, right after
// spawning the process; until it is answered every other request fails
// with not_initialized.
type HelloParams struct {
	Protocol int    `json:"protocol"`           // Version
	App      string `json:"app"`                // app version, for logs
	LogLevel string `json:"logLevel,omitempty"` // "debug" | "info" (default) | "warning" | "error"
}

// HelloResult describes the helper. The app refuses (and reports) a
// helper whose Protocol differs from Version, and hides what needs a
// method the helper does not list.
type HelloResult struct {
	Protocol int      `json:"protocol"`
	Version  string   `json:"version"` // helper build version
	PID      int      `json:"pid"`
	Methods  []string `json:"methods"` // requests it implements
	Device   Device   `json:"device"`
}

// Device is the machine the helper runs on.
type Device struct {
	Name string `json:"name"` // "MacBook Air (M2)"
	OS   string `json:"os"`   // "macOS 26.7"
}

// CancelParams names the job (the id of its request) to cancel.
type CancelParams struct {
	Job string `json:"job"`
}

// CancelResult says whether a running job was found. The job's own
// request then answers with a cancelled error.
type CancelResult struct {
	Cancelled bool `json:"cancelled"`
}

// Empty is the params/result of methods that take or return nothing ({}).
type Empty struct{}

// LogParams is a structured log line.
type LogParams struct {
	Level   string `json:"level"` // "debug" | "info" | "warning" | "error"
	Message string `json:"message"`
	Job     string `json:"job,omitempty"`
}

// ParamsFor returns a new value of the params type of method, or nil for
// an unknown method. ResultFor does the same for results. The fake helper
// and the fixture tests use them.
func ParamsFor(method string) any {
	switch method {
	case MethodHello:
		return new(HelloParams)
	case MethodShutdown, MethodCaptureDevices, MethodPermissionsStatus, MethodHotkeyStatus, MethodAppFrontmost, MethodTextFocus:
		return new(Empty)
	case MethodJobCancel:
		return new(CancelParams)
	case MethodPermissionsRequest:
		return new(PermissionsRequestParams)
	case MethodCaptureStart:
		return new(CaptureParams)
	case MethodCaptureStop:
		return new(CaptureStopParams)
	case MethodHotkeyConfigure:
		return new(HotkeyConfigureParams)
	case MethodTextSelection:
		return new(SelectionParams)
	case MethodTextPaste:
		return new(PasteParams)
	case MethodBrowserURL:
		return new(BrowserURLParams)
	case MethodAppList:
		return new(AppListParams)
	case MethodAppConfigure:
		return new(AppConfigureParams)
	case MethodAppIcons:
		return new(AppIconsParams)
	case MethodOutputWatch:
		return new(OutputWatchParams)
	case MethodOutputMute:
		return new(OutputMuteParams)
	case MethodOutputRelease:
		return new(OutputReleaseParams)
	case MethodOutputRecover:
		return new(OutputRecoverParams)
	case MethodOutputChanged:
		return new(OutputState)
	case MethodOutputReleased:
		return new(OutputReleasedParams)
	case MethodCaptureStarted:
		return new(CaptureStartedParams)
	case MethodCaptureLevel:
		return new(LevelParams)
	case MethodCapturePCM:
		return new(PCMParams)
	case MethodCaptureRestarted:
		return new(RestartedParams)
	case MethodHotkeyEvent:
		return new(HotkeyEventParams)
	case MethodHotkeyState:
		return new(HotkeyStatus)
	case MethodAppActivated:
		return new(App)
	case MethodLog:
		return new(LogParams)
	}
	return nil
}

// ResultFor: see ParamsFor.
func ResultFor(method string) any {
	switch method {
	case MethodHello:
		return new(HelloResult)
	case MethodShutdown, MethodAppConfigure:
		return new(Empty)
	case MethodJobCancel:
		return new(CancelResult)
	case MethodPermissionsStatus, MethodPermissionsRequest:
		return new(PermissionsResult)
	case MethodCaptureDevices:
		return new(DevicesResult)
	case MethodCaptureStart:
		return new(CaptureResult)
	case MethodCaptureStop:
		return new(CaptureStopResult)
	case MethodHotkeyConfigure, MethodHotkeyStatus:
		return new(HotkeyStatus)
	case MethodAppFrontmost:
		return new(App)
	case MethodTextFocus:
		return new(FocusResult)
	case MethodTextSelection:
		return new(SelectionResult)
	case MethodTextPaste:
		return new(PasteResult)
	case MethodBrowserURL:
		return new(BrowserURLResult)
	case MethodAppList:
		return new(AppListResult)
	case MethodAppIcons:
		return new(AppIconsResult)
	case MethodOutputWatch:
		return new(OutputState)
	case MethodOutputMute:
		return new(OutputMuteResult)
	case MethodOutputRelease:
		return new(OutputReleaseResult)
	case MethodOutputRecover:
		return new(OutputRecoverResult)
	}
	return nil
}

// Requests lists the requests of the protocol, Notifications its
// notifications.
var (
	Requests = []string{
		MethodHello, MethodShutdown, MethodJobCancel,
		MethodPermissionsStatus, MethodPermissionsRequest,
		MethodCaptureDevices, MethodCaptureStart, MethodCaptureStop,
		MethodHotkeyConfigure, MethodHotkeyStatus,
		MethodAppFrontmost, MethodAppList, MethodAppConfigure, MethodAppIcons,
		MethodTextFocus, MethodTextSelection, MethodTextPaste,
		MethodBrowserURL,
		MethodOutputWatch, MethodOutputMute, MethodOutputRelease, MethodOutputRecover,
	}
	Notifications = []string{
		MethodCaptureStarted, MethodCaptureLevel, MethodCapturePCM, MethodCaptureRestarted,
		MethodHotkeyEvent, MethodHotkeyState, MethodAppActivated, MethodOutputChanged, MethodOutputReleased, MethodLog,
	}
)

// IsJob reports whether method is a long-running, cancellable job.
func IsJob(method string) bool { return method == MethodCaptureStart }
