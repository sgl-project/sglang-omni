package proto

// ---- permissions.* ----

// PermissionsResult answers permissions.status and permissions.request.
// macOS checks the helper's permissions against VoxtNew.app, its
// responsible process: they are the app's.
type PermissionsResult struct {
	Microphone    string `json:"microphone"`    // a Perm* state
	Accessibility bool   `json:"accessibility"` // AXIsProcessTrusted: the event tap, AX reads, posting keys
}

// Microphone states.
const (
	PermGranted       = "granted"
	PermDenied        = "denied"
	PermNotDetermined = "notDetermined"
	PermRestricted    = "restricted"
)

// PermissionsRequestParams asks for one permission: the microphone shows
// macOS's prompt once (answered when the user chose); accessibility shows
// the prompt that adds VoxtNew to the list in System Settings (answered
// at once: the user grants it there).
type PermissionsRequestParams struct {
	Permission string `json:"permission"` // PermissionMicrophone | PermissionAccessibility
}

// Permissions, as in permissions.request and permission_denied's data.
const (
	PermissionMicrophone    = "microphone"
	PermissionAccessibility = "accessibility"
)

// ---- app.* ----

// App is a running app (app.frontmost, app.activated, a hotkey's front).
type App struct {
	PID      int    `json:"pid"`
	BundleID string `json:"bundleId,omitempty"`
	Name     string `json:"name,omitempty"`
}

// ---- app.configure ----

// AppConfigureParams says which apps the helper gives Chromium's
// AXManualAccessibility as they come to the front (once per process),
// besides the Electron apps it finds itself: Chromium builds a page's
// accessibility tree only when asked, and only a moment later, so
// browser.url and textio find it when a shortcut comes. The app sends its
// Chromium browsers (internal/appctx) while app enhancement is on; it
// replaces the last list, and a new helper has none. The result is Empty.
type AppConfigureParams struct {
	ManualAccessibility []string `json:"manualAccessibility,omitempty"` // bundle ids
}

// ---- app.list ----

// AppListParams lists apps: the running ones (regular apps, with a Dock
// icon), and with Installed also those in /Applications,
// /System/Applications and ~/Applications (one level of folders down).
type AppListParams struct {
	Installed bool `json:"installed,omitempty"`
}

// AppListResult lists apps by name, each bundle id once.
type AppListResult struct {
	Apps []AppInfo `json:"apps"`
}

// AppInfo is an app the user may pick (App enhancement's members).
type AppInfo struct {
	BundleID string `json:"bundleId"`
	Name     string `json:"name"`
	Running  bool   `json:"running"`
	PID      int    `json:"pid,omitempty"`  // when running
	Path     string `json:"path,omitempty"` // its bundle
}

// ---- app.icons ----

// AppIconsParams asks for the icons of apps, as the Finder shows them:
// each by its bundle's Path when the app has it (app.list's), else the
// bundle the system finds for BundleID. At most MaxIcons apps a request;
// Size is the PNG's side in pixels (default 64, at most 256: 32 points at
// 2×).
type AppIconsParams struct {
	Apps []AppRef `json:"apps"`
	Size int      `json:"size,omitempty"`
}

// AppRef names an app for app.icons.
type AppRef struct {
	BundleID string `json:"bundleId"`
	Path     string `json:"path,omitempty"`
}

// AppIconsResult has an icon per app asked for, in its order.
type AppIconsResult struct {
	Icons []AppIcon `json:"icons"`
}

// AppIcon is an app's icon: PNG is absent when no bundle was found (the
// app was removed), and the app shows a generic icon.
type AppIcon struct {
	BundleID string `json:"bundleId"`
	PNG      []byte `json:"png,omitempty"` // base64 on the wire
}

// Icon sizes: the default and the largest side app.icons draws, and the
// most apps a request may name (a full answer stays far below MaxLine).
const (
	IconSize    = 64
	MaxIconSize = 256
	MaxIcons    = 48
)
