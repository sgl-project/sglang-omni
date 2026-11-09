//go:build linux && (amd64 || arm64)

package gamepad

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

// Linux reads pads from evdev, /dev/input/event*, which systemd-logind
// lets the user at the seat read. The kernel's drivers report gamepads in
// one layout (Documentation/input/gamepad.rst): BTN_SOUTH and the rest by
// place, sticks on ABS_X/Y and ABS_RX/RY, triggers on ABS_Z/RZ, the
// d-pad on a hat or buttons. Xbox controllers (xpad, and Steam's virtual
// pad) are the exception that predates it: they report X as BTN_X, the
// code of BTN_NORTH, and Y as BTN_Y, that of BTN_WEST.

// Event types and codes.
const (
	evKey = 0x01
	evAbs = 0x03

	btnSouth  = 0x130
	btnEast   = 0x131
	btnNorth  = 0x133 // also BTN_X
	btnWest   = 0x134 // also BTN_Y
	btnTL     = 0x136
	btnTR     = 0x137
	btnTL2    = 0x138
	btnTR2    = 0x139
	btnSelect = 0x13a
	btnStart  = 0x13b
	btnMode   = 0x13c
	btnThumbL = 0x13d
	btnThumbR = 0x13e
	btnUp     = 0x220
	btnDown   = 0x221
	btnLeft   = 0x222
	btnRight  = 0x223

	absX     = 0x00
	absY     = 0x01
	absZ     = 0x02
	absRX    = 0x03
	absRY    = 0x04
	absRZ    = 0x05
	absHat0X = 0x10
	absHat0Y = 0x11
	absMax   = 0x3f
	keyMax   = 0x2ff
)

// ioctl requests of evdev: _IOC(dir, 'E', nr, size).
func ioc(dir, nr, size uintptr) uintptr { return dir<<30 | size<<16 | 'E'<<8 | nr }

const iocRead = 2

func eviocgname(n uintptr) uintptr    { return ioc(iocRead, 0x06, n) }
func eviocgbit(ev, n uintptr) uintptr { return ioc(iocRead, 0x20+ev, n) }
func eviocgabs(abs uintptr) uintptr   { return ioc(iocRead, 0x40+abs, unsafe.Sizeof(absInfo{})) }
func eviocgkey(n uintptr) uintptr     { return ioc(iocRead, 0x18, n) }
func eviocgid() uintptr               { return ioc(iocRead, 0x02, unsafe.Sizeof(inputID{})) }
func ioctl(fd int, req uintptr, p unsafe.Pointer) bool {
	_, _, e := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), req, uintptr(p))
	return e == 0
}

type absInfo struct{ value, minimum, maximum, fuzz, flat, resolution int32 }

type inputID struct{ bustype, vendor, product, version uint16 }

type evdevPad struct {
	path  string
	fd    int
	title string
	xbox  bool // BTN_X and BTN_Y are X and Y, west and north
	// rightOnZ: a pad of no ABS_RX has its right stick on ABS_Z/RZ, and
	// triggers only as buttons.
	rightOnZ bool
	abs      [absMax + 1]absInfo
	hasAbs   [absMax + 1]bool
	keys     [keyMax + 1]bool
	axes     [absMax + 1]int32
	buf      []byte
}

func scan(known map[string]device) []device {
	paths, _ := filepath.Glob("/dev/input/event*")
	var found []device
	for _, p := range paths {
		if k, ok := known[p]; ok {
			found = append(found, k)
			continue
		}
		if d := openEvdev(p); d != nil {
			found = append(found, d)
		}
	}
	return found
}

func bit(b []byte, i int) bool { return i/8 < len(b) && b[i/8]&(1<<(i%8)) != 0 }

// openEvdev opens the device at path if it is a gamepad.
func openEvdev(path string) *evdevPad {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NONBLOCK|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil
	}
	keys := make([]byte, keyMax/8+1)
	if !ioctl(fd, eviocgbit(evKey, uintptr(len(keys))), unsafe.Pointer(&keys[0])) || !bit(keys, btnSouth) {
		syscall.Close(fd)
		return nil
	}
	d := &evdevPad{path: path, fd: fd, buf: make([]byte, 24*64)}
	name := make([]byte, 256)
	if ioctl(fd, eviocgname(uintptr(len(name))), unsafe.Pointer(&name[0])) {
		d.title = strings.TrimRight(string(name), "\x00")
	}
	var id inputID
	ioctl(fd, eviocgid(), unsafe.Pointer(&id))
	d.xbox = id.vendor == 0x045e || id.vendor == 0x28de || driver(path) == "xpad"
	abs := make([]byte, absMax/8+1)
	ioctl(fd, eviocgbit(evAbs, uintptr(len(abs))), unsafe.Pointer(&abs[0]))
	for a := range absMax + 1 {
		if bit(abs, a) && ioctl(fd, eviocgabs(uintptr(a)), unsafe.Pointer(&d.abs[a])) {
			d.hasAbs[a] = true
			d.axes[a] = d.abs[a].value
		}
	}
	d.rightOnZ = !d.hasAbs[absRX] && d.hasAbs[absZ]
	state := make([]byte, keyMax/8+1)
	if ioctl(fd, eviocgkey(uintptr(len(state))), unsafe.Pointer(&state[0])) {
		for k := range keyMax + 1 {
			d.keys[k] = bit(state, k)
		}
	}
	return d
}

// driver returns the name of the kernel driver of an evdev device.
func driver(path string) string {
	link, err := os.Readlink("/sys/class/input/" + filepath.Base(path) + "/device/device/driver")
	if err != nil {
		return ""
	}
	return filepath.Base(link)
}

func (d *evdevPad) key() string  { return d.path }
func (d *evdevPad) name() string { return d.title }
func (d *evdevPad) close()       { syscall.Close(d.fd) }

// drain reads the events waiting, and reports false once the device is
// gone.
func (d *evdevPad) drain() bool {
	for {
		n, err := syscall.Read(d.fd, d.buf)
		if err == syscall.EAGAIN || err == syscall.EINTR {
			return true
		}
		if err != nil || n == 0 {
			return false // ENODEV: unplugged
		}
		for i := 0; i+24 <= n; i += 24 {
			e := d.buf[i:]
			typ := binary.LittleEndian.Uint16(e[16:])
			code := binary.LittleEndian.Uint16(e[18:])
			value := int32(binary.LittleEndian.Uint32(e[20:]))
			switch {
			case typ == evKey && int(code) <= keyMax:
				d.keys[code] = value != 0
			case typ == evAbs && int(code) <= absMax:
				d.axes[code] = value
			}
		}
		if n < len(d.buf) {
			return true
		}
	}
}

// norm maps axis a to -1..1, or 0..1 when half.
func (d *evdevPad) norm(a int, half bool) float32 {
	if !d.hasAbs[a] {
		return 0
	}
	lo, hi := float32(d.abs[a].minimum), float32(d.abs[a].maximum)
	if hi <= lo {
		return 0
	}
	t := (float32(d.axes[a]) - lo) / (hi - lo)
	if half {
		return min(max(t, 0), 1)
	}
	return min(max(t*2-1, -1), 1)
}

func (d *evdevPad) read(s *State) bool {
	if !d.drain() {
		return false
	}
	press := func(b Button, code int) {
		if d.keys[code] {
			s.Buttons[b] = 1
		}
	}
	press(South, btnSouth)
	press(East, btnEast)
	if d.xbox {
		press(West, btnNorth)
		press(North, btnWest)
	} else {
		press(North, btnNorth)
		press(West, btnWest)
	}
	press(LeftShoulder, btnTL)
	press(RightShoulder, btnTR)
	press(Back, btnSelect)
	press(Start, btnStart)
	press(Guide, btnMode)
	press(LeftStickButton, btnThumbL)
	press(RightStickButton, btnThumbR)
	press(Up, btnUp)
	press(Down, btnDown)
	press(Left, btnLeft)
	press(Right, btnRight)
	if d.hasAbs[absHat0X] {
		if v := d.axes[absHat0X]; v < 0 {
			s.Buttons[Left] = 1
		} else if v > 0 {
			s.Buttons[Right] = 1
		}
		if v := d.axes[absHat0Y]; v < 0 {
			s.Buttons[Up] = 1
		} else if v > 0 {
			s.Buttons[Down] = 1
		}
	}
	s.Axes[LeftX], s.Axes[LeftY] = d.norm(absX, false), d.norm(absY, false)
	if d.rightOnZ {
		s.Axes[RightX], s.Axes[RightY] = d.norm(absZ, false), d.norm(absRZ, false)
	} else {
		s.Axes[RightX], s.Axes[RightY] = d.norm(absRX, false), d.norm(absRY, false)
		s.Buttons[LeftTrigger] = d.norm(absZ, true)
		s.Buttons[RightTrigger] = d.norm(absRZ, true)
	}
	press(LeftTrigger, btnTL2)
	press(RightTrigger, btnTR2)
	return true
}
