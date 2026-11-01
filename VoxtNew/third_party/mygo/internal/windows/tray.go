//go:build windows && (amd64 || arm64)

package windows

import (
	"bytes"
	"errors"
	"fmt"
	"image/png"
	"unsafe"

	"github.com/egoist/mygo/internal/accelerator"
	"github.com/egoist/mygo/internal/platform"
)

// notifyIconData is NOTIFYICONDATAW.
type notifyIconData struct {
	Size            uint32
	Wnd             uintptr
	ID              uint32
	Flags           uint32
	CallbackMessage uint32
	Icon            uintptr
	Tip             [128]uint16
	State           uint32
	StateMask       uint32
	Info            [256]uint16
	Version         uint32 // union with uTimeout
	InfoTitle       [64]uint16
	InfoFlags       uint32
	GUIDItem        GUID
	BalloonIcon     uintptr
}

const (
	nimAdd    = 0
	nimModify = 1
	nimDelete = 2

	nifMessage = 0x01
	nifIcon    = 0x02
	nifTip     = 0x04
	nifInfo    = 0x10
	nifShowTip = 0x80

	niifNone    = 0x00
	niifUser    = 0x04
	niifNoSound = 0x10

	ninBalloonTimeout   = wmUser + 4
	ninBalloonUserClick = wmUser + 5
)

func copyUTF16(dst []uint16, s string) {
	src := utf16Units(s)
	n := copy(dst[:len(dst)-1], src)
	dst[n] = 0
}

type tray struct {
	b     *Backend
	h     platform.TrayHandler
	id    uint32
	icon  uintptr
	tip   string
	menu  *platform.Menu
	added bool
	// notification is the id of the notification the icon shows, if any.
	notification string
	temporary    bool // added only to show a notification
}

func (b *Backend) NewTray(h platform.TrayHandler) (platform.Tray, error) {
	b.nextTray++
	t := &tray{b: b, h: h, id: b.nextTray}
	b.trays[t.id] = t
	t.add()
	return t, nil
}

func (t *tray) data(flags uint32) notifyIconData {
	d := notifyIconData{Wnd: t.b.appHwnd, ID: t.id, Flags: flags | nifMessage, CallbackMessage: wmAppTray, Icon: t.icon}
	d.Size = uint32(unsafe.Sizeof(d))
	if t.icon != 0 {
		d.Flags |= nifIcon
	}
	if t.tip != "" {
		d.Flags |= nifTip | nifShowTip
		copyUTF16(d.Tip[:], t.tip)
	}
	return d
}

func (t *tray) add() {
	d := t.data(0)
	procShellNotifyIconW.Call(nimAdd, uintptr(unsafe.Pointer(&d)))
	t.added = true
}

func (t *tray) update() {
	d := t.data(0)
	procShellNotifyIconW.Call(nimModify, uintptr(unsafe.Pointer(&d)))
}

func (t *tray) SetImage(data []byte, template bool) error {
	icon, err := iconFromPNG(data, 0)
	if err != nil {
		return err
	}
	if t.icon != 0 {
		procDestroyIcon.Call(t.icon)
	}
	t.icon = icon
	t.update()
	return nil
}

func (t *tray) SetTitle(string) {}

func (t *tray) SetToolTip(tip string) {
	t.tip = tip
	t.update()
}

func (t *tray) SetMenu(m *platform.Menu) { t.menu = m }

func (t *tray) PopUpMenu(m *platform.Menu) {
	if m == nil {
		m = t.menu
	}
	if m == nil {
		return
	}
	id := t.b.menus.newOwner()
	defer t.b.menus.drop(id)
	menu := t.b.buildMenu(m, false, id, nil)
	defer procDestroyMenu.Call(menu)
	var pt point
	procGetCursorPos.Call(uintptr(unsafe.Pointer(&pt)))
	if cmd := t.b.trackPopup(menu, t.b.appHwnd, pt, 0); cmd != 0 {
		t.b.menuCommand(cmd, nil)
	}
}

func (t *tray) Bounds() platform.Rect {
	r, ok := t.rect()
	if !ok {
		return platform.Rect{}
	}
	dpi := monitorDPI(monitorAt(int(r.Left), int(r.Top)))
	return platform.Rect{X: toDIP(r.Left, dpi), Y: toDIP(r.Top, dpi), Width: toDIP(r.Right-r.Left, dpi), Height: toDIP(r.Bottom-r.Top, dpi)}
}

// rect returns where the icon is on the screen, in physical pixels.
func (t *tray) rect() (rect, bool) {
	type identifier struct {
		Size uint32
		Wnd  uintptr
		ID   uint32
		GUID GUID
	}
	ident := identifier{Wnd: t.b.appHwnd, ID: t.id}
	ident.Size = uint32(unsafe.Sizeof(ident))
	var r rect
	if !has(procShellNotifyIconGetRect) {
		return r, false
	}
	if hr, _, _ := procShellNotifyIconGetRect.Call(uintptr(unsafe.Pointer(&ident)), uintptr(unsafe.Pointer(&r))); failed(hr) {
		return r, false
	}
	return r, r.Right > r.Left && r.Bottom > r.Top
}

func (t *tray) Destroy() {
	if t.added {
		d := t.data(0)
		procShellNotifyIconW.Call(nimDelete, uintptr(unsafe.Pointer(&d)))
		t.added = false
	}
	if t.icon != 0 && !t.temporary {
		procDestroyIcon.Call(t.icon)
		t.icon = 0
	}
	delete(t.b.trays, t.id)
}

// trayMessage handles the callback message of the notification area.
func (b *Backend) trayMessage(id uint32, event uint32) {
	t := b.trays[id]
	if t == nil {
		return
	}
	switch event {
	case wmLButtonUp:
		if t.h != nil {
			t.h.Clicked()
		}
	case wmRButtonUp:
		if t.h != nil {
			t.h.RightClicked()
		}
		if t.menu != nil {
			t.PopUpMenu(t.menu)
		}
	case ninBalloonUserClick:
		if n := t.notification; n != "" {
			b.finishNotification(t)
			b.h.NotificationClicked(n)
		}
	case ninBalloonTimeout:
		b.finishNotification(t)
	}
}

// iconFromPNG makes an icon of a PNG image (size 0: the image's size).
func iconFromPNG(data []byte, size int) (uintptr, error) {
	cfg, err := png.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return 0, err
	}
	if size == 0 {
		size = cfg.Width
	}
	icon, _, _ := procCreateIconFromResourceEx.Call(uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)), 1, 0x00030000, uintptr(size), uintptr(size), 0)
	if icon == 0 {
		return 0, errors.New("mygo: cannot create an icon from the image")
	}
	return icon, nil
}

// Notifications are notification area balloons, which Windows 10 and later
// show as toasts. They need an icon: the first tray icon, else a temporary
// one with the application's icon.

type notification struct{ tray *tray }

func (b *Backend) NotificationsSupported() bool { return true }

func (b *Backend) ShowNotification(n *platform.Notification, done func(error)) {
	done(b.showNotification(n))
}

func (b *Backend) showNotification(n *platform.Notification) error {
	var t *tray
	for _, x := range b.trays {
		if !x.temporary && x.notification == "" {
			t = x
			break
		}
	}
	if t == nil {
		b.nextTray++
		t = &tray{b: b, id: b.nextTray, temporary: true}
		t.icon, _, _ = procLoadIconW.Call(instance(), 1)
		if t.icon == 0 {
			t.icon, _, _ = procLoadIconW.Call(0, 32512) // IDI_APPLICATION
		}
		b.trays[t.id] = t
		t.add()
	}
	t.notification = n.ID
	b.notifications[n.ID] = &notification{tray: t}
	d := t.data(nifInfo)
	title := n.Title
	body := n.Body
	if n.Subtitle != "" {
		body = n.Subtitle + "\n" + body
	}
	copyUTF16(d.InfoTitle[:], title)
	copyUTF16(d.Info[:], body)
	d.InfoFlags = niifUser
	if n.Silent {
		d.InfoFlags |= niifNoSound
	}
	if r, _, _ := procShellNotifyIconW.Call(nimModify, uintptr(unsafe.Pointer(&d))); r == 0 {
		b.finishNotification(t)
		return fmt.Errorf("mygo: cannot show the notification")
	}
	return nil
}

func (b *Backend) RemoveNotification(id string) {
	if n := b.notifications[id]; n != nil {
		b.finishNotification(n.tray)
	}
}

func (b *Backend) RemoveAllNotifications() {
	for _, n := range b.notifications {
		b.finishNotification(n.tray)
	}
}

func (b *Backend) finishNotification(t *tray) {
	delete(b.notifications, t.notification)
	t.notification = ""
	if t.temporary {
		t.Destroy()
		return
	}
	d := t.data(nifInfo) // an empty balloon hides the current one
	procShellNotifyIconW.Call(nimModify, uintptr(unsafe.Pointer(&d)))
}

// Global shortcuts.

func (b *Backend) RegisterHotkey(id int, acc string) error {
	a, err := accelerator.Parse(acc, "windows")
	if err != nil {
		return err
	}
	vk, ok := virtualKey(a.Key)
	if !ok {
		return fmt.Errorf("mygo: unsupported key in %q", acc)
	}
	const (
		modAlt, modControl, modShift, modWin, modNoRepeat = 0x1, 0x2, 0x4, 0x8, 0x4000
	)
	mods := uintptr(modNoRepeat)
	if a.Has(accelerator.Alt) {
		mods |= modAlt
	}
	if a.Has(accelerator.Ctrl) {
		mods |= modControl
	}
	if a.Has(accelerator.Shift) {
		mods |= modShift
	}
	if a.Has(accelerator.Super) {
		mods |= modWin
	}
	if r, _, err := procRegisterHotKey.Call(b.appHwnd, uintptr(id), mods, uintptr(vk)); r == 0 {
		return fmt.Errorf("mygo: cannot register %q: %v", acc, err)
	}
	return nil
}

func (b *Backend) UnregisterHotkey(id int) { procUnregisterHotKey.Call(b.appHwnd, uintptr(id)) }
