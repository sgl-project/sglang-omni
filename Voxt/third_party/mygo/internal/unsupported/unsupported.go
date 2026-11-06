// Package unsupported is the backend for platforms without a native
// implementation. Init fails, everything else is inert.
package unsupported

import (
	"errors"
	"runtime"
	"time"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

// Backend reports that the platform is not supported.
type Backend struct{}

// New returns the unsupported backend.
func New() *Backend { return &Backend{} }

var errUnsupported = errors.New("mygo: " + runtime.GOOS + "/" + runtime.GOARCH + " is not supported yet")

func (*Backend) Name() string                                        { return "unsupported" }
func (*Backend) Init(platform.AppHandler, platform.AppOptions) error { return errUnsupported }
func (*Backend) Run() error                                          { return errUnsupported }
func (*Backend) Quit()                                               {}
func (*Backend) IsMainThread() bool                                  { return true }
func (*Backend) Signal()                                             {}
func (*Backend) Step()                                               {}
func (*Backend) Wake()                                               {}
func (*Backend) NewWindow(*platform.WindowOptions, platform.WindowHandler) (platform.Window, error) {
	return nil, errUnsupported
}
func (*Backend) SetApplicationMenu(*platform.Menu)                          {}
func (*Backend) PopupMenu(*platform.Menu, platform.Window, *platform.Point) {}
func (*Backend) UpdateMenuItem(*platform.MenuItem)                          {}
func (*Backend) App() platform.AppController                                { return app{} }
func (*Backend) Dialogs() platform.Dialogs                                  { return dialogs{} }
func (*Backend) Clipboard() platform.Clipboard                              { return clipboard{} }
func (*Backend) Shell() platform.Shell                                      { return shell{} }
func (*Backend) Screen() platform.Screen                                    { return screen{} }
func (*Backend) Theme() platform.Theme                                      { return theme{} }
func (*Backend) Power() platform.Power                                      { return power{} }
func (*Backend) Network() platform.Network                                  { return network{} }
func (*Backend) Secrets() platform.Secrets                                  { return secrets{} }
func (*Backend) NewTray(platform.TrayHandler) (platform.Tray, error)        { return nil, errUnsupported }
func (*Backend) RegisterHotkey(int, string) error                           { return errUnsupported }
func (*Backend) UnregisterHotkey(int)                                       {}
func (*Backend) NotificationsSupported() bool                               { return false }
func (*Backend) WebViewAvailable() bool                                     { return false }
func (*Backend) RemoveNotification(string)                                  {}

func (*Backend) ShowNotification(_ *platform.Notification, done func(error)) {
	done(errUnsupported)
}

func (*Backend) RemoveAllNotifications() {}

type app struct{}

func (app) SetActivationPolicy(string)                        {}
func (app) Activate()                                         {}
func (app) Hide()                                             {}
func (app) Unhide()                                           {}
func (app) IsHidden() bool                                    { return false }
func (app) SetBadge(string)                                   {}
func (app) Badge() string                                     { return "" }
func (app) Bounce(bool) int                                   { return 0 }
func (app) CancelBounce(int)                                  {}
func (app) SetDockIcon([]byte) error                          { return errUnsupported }
func (app) ShowAboutPanel(platform.AboutPanelOptions)         {}
func (app) PreferredLanguages() []string                      { return []string{"en-US"} }
func (app) Region() platform.Region                           { return platform.Region{FirstWeekday: -1} }
func (app) SavedLocale() (string, bool)                       { return "", false }
func (app) SaveLocale(string) bool                            { return false }
func (app) SetLanguages([]string, bool)                       {}
func (app) Package() (platform.PackageInfo, bool)             { return platform.PackageInfo{}, false }
func (app) RegisterURLScheme(string, string, string) error    { return errUnsupported }
func (app) UnregisterURLScheme(string, string, string) error  { return errUnsupported }
func (app) IsURLSchemeRegistered(string, string, string) bool { return false }
func (app) SetOpenAtLogin(bool, string, string, string) error { return errUnsupported }
func (app) OpenAtLogin(string, string, string) bool           { return false }
func (app) OpenedAtLogin() bool                               { return false }
func (app) SetDockMenu(*platform.Menu)                        {}
func (app) ClearBrowsingData(done func(error))                { done(errUnsupported) }
func (app) AddRecentDocument(string) error                    { return errUnsupported }
func (app) ClearRecentDocuments() error                       { return errUnsupported }
func (app) SetJumpList(platform.JumpList) error               { return errUnsupported }

func (app) ListCookies(_ string, done func([]platform.Cookie, error)) {
	done(nil, platform.ErrUnsupported)
}
func (app) SetCookie(_ platform.Cookie, done func(error))       { done(platform.ErrUnsupported) }
func (app) DeleteCookie(_ platform.CookieKey, done func(error)) { done(platform.ErrUnsupported) }
func (app) ClearCookies(done func(error))                       { done(platform.ErrUnsupported) }

type dialogs struct{}

func (dialogs) ShowOpenDialog(_ platform.Window, _ *platform.OpenDialogOptions, cb func([]string, error)) {
	cb(nil, errUnsupported)
}
func (dialogs) ShowSaveDialog(_ platform.Window, _ *platform.SaveDialogOptions, cb func(string, error)) {
	cb("", errUnsupported)
}
func (dialogs) ShowMessageBox(_ platform.Window, _ *platform.MessageBoxOptions, cb func(platform.MessageBoxResult, error)) {
	cb(platform.MessageBoxResult{}, errUnsupported)
}

type clipboard struct{}

func (clipboard) ReadData([]transfer.Format) (transfer.Data, error) {
	return transfer.Data{}, errUnsupported
}
func (clipboard) WriteData(transfer.Data, func()) error { return errUnsupported }
func (clipboard) Formats() []transfer.Format            { return nil }
func (clipboard) Flush() error                          { return errUnsupported }
func (clipboard) Close()                                {}

func (clipboard) ReadText() string           { return "" }
func (clipboard) WriteText(string)           {}
func (clipboard) ReadHTML() string           { return "" }
func (clipboard) WriteHTML(string)           {}
func (clipboard) ReadImage() []byte          { return nil }
func (clipboard) WriteImage([]byte) error    { return errUnsupported }
func (clipboard) Clear()                     {}
func (clipboard) AvailableFormats() []string { return nil }

type shell struct{}

func (shell) OpenExternal(string) error            { return errUnsupported }
func (shell) OpenPath(string) error                { return errUnsupported }
func (shell) ShowItemInFolder(string)              {}
func (shell) TrashItem(string) error               { return errUnsupported }
func (shell) Beep()                                {}
func (shell) FileIcon(string, int) ([]byte, error) { return nil, errUnsupported }
func (shell) OpenApplication(string) error         { return errUnsupported }

func (*Backend) Applications() ([]platform.Application, error) { return nil, errUnsupported }

type screen struct{}

func (screen) Displays() []platform.Display { return nil }
func (screen) CursorPoint() platform.Point  { return platform.Point{} }

type theme struct{}

func (theme) IsDark() bool     { return false }
func (theme) SetSource(string) {}
func (theme) UIFont() string   { return "" }

func (theme) FontRendering() platform.FontRendering { return platform.FontRendering{} }

func (theme) Preferences() platform.Preferences { return platform.Preferences{} }

type power struct{}

func (power) Watch()                        {}
func (power) KeepAwake(bool, string) func() { return func() {} }
func (power) OnBattery() bool               { return false }
func (power) IdleTime() time.Duration       { return 0 }

type network struct{}

func (network) Watch()                         {}
func (network) Status() platform.NetworkStatus { return platform.NetworkStatus{} }

type secrets struct{}

func (secrets) SetSecret(string, string, []byte) error { return platform.ErrUnsupported }
func (secrets) Secret(string, string) ([]byte, error)  { return nil, platform.ErrUnsupported }
func (secrets) DeleteSecret(string, string) error      { return platform.ErrUnsupported }
