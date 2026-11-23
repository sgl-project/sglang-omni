//go:build windows && (amd64 || arm64)

package gamepad

import (
	"strconv"
	"sync"
	"syscall"
	"unsafe"
)

// Windows reads pads through XInput, which has four slots, for Xbox
// controllers and those that act as one. XInputGetStateEx, exported by
// ordinal 100, reads the Guide button too.

type xinputGamepad struct {
	buttons      uint16
	leftTrigger  uint8
	rightTrigger uint8
	thumbLX      int16
	thumbLY      int16
	thumbRX      int16
	thumbRY      int16
}

type xinputState struct {
	packet  uint32
	gamepad xinputGamepad
}

const (
	errorSuccess        = 0
	xinputSlots         = 4
	xinputDpadUp        = 0x0001
	xinputDpadDown      = 0x0002
	xinputDpadLeft      = 0x0004
	xinputDpadRight     = 0x0008
	xinputStart         = 0x0010
	xinputBack          = 0x0020
	xinputLeftThumb     = 0x0040
	xinputRightThumb    = 0x0080
	xinputLeftShoulder  = 0x0100
	xinputRightShoulder = 0x0200
	xinputGuide         = 0x0400
	xinputA             = 0x1000
	xinputB             = 0x2000
	xinputX             = 0x4000
	xinputY             = 0x8000
)

var (
	xinputOnce     sync.Once
	xinputGetState uintptr
)

func loadXInput() {
	xinputOnce.Do(func() {
		kernel32 := syscall.NewLazyDLL("kernel32.dll")
		getProc := kernel32.NewProc("GetProcAddress")
		for _, name := range []string{"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"} {
			lib, err := syscall.LoadLibrary(name)
			if err != nil {
				continue
			}
			// XInputGetStateEx, by ordinal, where there is one.
			if p, _, _ := getProc.Call(uintptr(lib), 100); p != 0 {
				xinputGetState = p
				return
			}
			if p, err := syscall.GetProcAddress(lib, "XInputGetState"); err == nil {
				xinputGetState = p
				return
			}
		}
	})
}

func getState(slot int, s *xinputState) bool {
	r, _, _ := syscall.SyscallN(xinputGetState, uintptr(slot), uintptr(unsafe.Pointer(s)))
	return r == errorSuccess
}

type xinputPad struct{ slot int }

func scan(known map[string]device) []device {
	loadXInput()
	if xinputGetState == 0 {
		return nil
	}
	var found []device
	for slot := range xinputSlots {
		d := &xinputPad{slot}
		if k, ok := known[d.key()]; ok {
			// Read as it is used; a slot that empties disconnects there.
			found = append(found, k)
			continue
		}
		// Asking an empty slot is slow, which is why scans are rare.
		var st xinputState
		if getState(slot, &st) {
			found = append(found, d)
		}
	}
	return found
}

func (d *xinputPad) key() string  { return "xinput" + strconv.Itoa(d.slot) }
func (d *xinputPad) name() string { return "Xbox Controller " + strconv.Itoa(d.slot+1) }
func (d *xinputPad) close()       {}

func (d *xinputPad) read(s *State) bool {
	var st xinputState
	if !getState(d.slot, &st) {
		return false
	}
	g := st.gamepad
	bits := [...]struct {
		b    Button
		mask uint16
	}{
		{South, xinputA}, {East, xinputB}, {West, xinputX}, {North, xinputY},
		{LeftShoulder, xinputLeftShoulder}, {RightShoulder, xinputRightShoulder},
		{Back, xinputBack}, {Start, xinputStart},
		{LeftStickButton, xinputLeftThumb}, {RightStickButton, xinputRightThumb},
		{Up, xinputDpadUp}, {Down, xinputDpadDown}, {Left, xinputDpadLeft}, {Right, xinputDpadRight},
		{Guide, xinputGuide},
	}
	for _, b := range bits {
		if g.buttons&b.mask != 0 {
			s.Buttons[b.b] = 1
		}
	}
	s.Buttons[LeftTrigger] = float32(g.leftTrigger) / 255
	s.Buttons[RightTrigger] = float32(g.rightTrigger) / 255
	// XInput's y axes go up.
	s.Axes[LeftX] = thumb(g.thumbLX)
	s.Axes[LeftY] = -thumb(g.thumbLY)
	s.Axes[RightX] = thumb(g.thumbRX)
	s.Axes[RightY] = -thumb(g.thumbRY)
	return true
}

func thumb(v int16) float32 { return max(float32(v)/32767, -1) }
