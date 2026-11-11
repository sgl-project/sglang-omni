package mygo

// Permission is something a page asks the user for.
type Permission string

// Permissions.
const (
	PermissionCamera        Permission = "camera"
	PermissionMicrophone    Permission = "microphone"
	PermissionGeolocation   Permission = "geolocation"
	PermissionNotifications Permission = "notifications"
)

// PermissionRequest is passed to the permission handler of a window.
type PermissionRequest struct {
	// Permissions asked for together, such as the camera and the
	// microphone of a video call.
	Permissions []Permission
	// Origin of the page asking, e.g. "https://example.com".
	Origin string
}

// SetPermissionHandler decides what the pages of the window may use, such
// as the camera: fn returns whether to grant a request, and runs on the main
// thread. nil restores the default: the app's own pages (see
// PageOptions.TrustedOrigins) get what they ask for, other pages do not.
//
// The operating system may still ask the user, like macOS does once per
// app for the camera and the microphone. That needs usage descriptions in
// macos.infoPlist of mygo.config.ts (NSCameraUsageDescription,
// NSMicrophoneUsageDescription), without which macOS ends the app.
func (p *Page) SetPermissionHandler(fn func(req PermissionRequest) bool) {
	p.w.mu.Lock()
	p.w.permissionHandler = fn
	p.w.mu.Unlock()
}

func (h *windowHandler) PermissionRequested(kinds []string, origin string) bool {
	h.w.mu.Lock()
	fn := h.w.permissionHandler
	h.w.mu.Unlock()
	if fn == nil {
		return h.w.isTrusted(origin)
	}
	req := PermissionRequest{Origin: origin}
	for _, k := range kinds {
		req.Permissions = append(req.Permissions, Permission(k))
	}
	return fn(req)
}
