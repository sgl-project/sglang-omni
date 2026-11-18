package gamepad

import (
	"runtime"
	"strconv"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

// macOS reads pads through GameController.framework: GCController's
// controllers with an extended gamepad profile, which Xbox, PlayStation,
// Switch Pro and MFi controllers have. The framework finds controllers
// on the main run loop, which a running MyGo app turns.

var (
	gcOnce sync.Once
	gcOK   bool

	poolPush, poolPop uintptr
	sendFloat         func(id objc.ID, sel objc.SEL) float32

	selControllers, selCount, selObjectAt, selExtended, selVendor, selUTF8,
	selValue, selXAxis, selYAxis, selRetain, selRelease, selResponds objc.SEL
	buttonSels                                                              [ButtonCount]objc.SEL
	selDpad, selLeftStick, selRightStick, selUp, selDown, selLeft, selRight objc.SEL
)

func loadGC() bool {
	gcOnce.Do(func() {
		if _, err := purego.Dlopen("/System/Library/Frameworks/GameController.framework/GameController", purego.RTLD_LAZY|purego.RTLD_GLOBAL); err != nil {
			return
		}
		libobjc, err := purego.Dlopen("/usr/lib/libobjc.A.dylib", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if err != nil {
			return
		}
		if poolPush, err = purego.Dlsym(libobjc, "objc_autoreleasePoolPush"); err != nil {
			return
		}
		if poolPop, err = purego.Dlsym(libobjc, "objc_autoreleasePoolPop"); err != nil {
			return
		}
		msgSend, err := purego.Dlsym(libobjc, "objc_msgSend")
		if err != nil {
			return
		}
		purego.RegisterFunc(&sendFloat, msgSend)
		sel := objc.RegisterName
		selControllers, selCount, selObjectAt = sel("controllers"), sel("count"), sel("objectAtIndex:")
		selExtended, selVendor, selUTF8 = sel("extendedGamepad"), sel("vendorName"), sel("UTF8String")
		selValue, selXAxis, selYAxis = sel("value"), sel("xAxis"), sel("yAxis")
		selRetain, selRelease, selResponds = sel("retain"), sel("release"), sel("respondsToSelector:")
		selDpad, selLeftStick, selRightStick = sel("dpad"), sel("leftThumbstick"), sel("rightThumbstick")
		selUp, selDown, selLeft, selRight = sel("up"), sel("down"), sel("left"), sel("right")
		for b, name := range map[Button]string{
			South: "buttonA", East: "buttonB", West: "buttonX", North: "buttonY",
			LeftShoulder: "leftShoulder", RightShoulder: "rightShoulder",
			LeftTrigger: "leftTrigger", RightTrigger: "rightTrigger",
			Back: "buttonOptions", Start: "buttonMenu", Guide: "buttonHome",
			LeftStickButton: "leftThumbstickButton", RightStickButton: "rightThumbstickButton",
		} {
			buttonSels[b] = sel(name)
		}
		gcOK = objc.GetClass("GCController") != 0
	})
	return gcOK
}

// withPool runs fn in an autorelease pool, for the objects the framework
// returns autoreleased. A pool belongs to its thread: the goroutine stays
// on one until the pool is popped.
func withPool(fn func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	p, _, _ := purego.SyscallN(poolPush)
	defer purego.SyscallN(poolPop, p)
	fn()
}

type gcPad struct {
	ctl   objc.ID // retained
	title string
}

func controllers() []objc.ID {
	list := objc.ID(objc.GetClass("GCController")).Send(selControllers)
	if list == 0 {
		return nil
	}
	n := int(objc.Send[uint](list, selCount))
	ids := make([]objc.ID, n)
	for i := range n {
		ids[i] = list.Send(selObjectAt, uint(i))
	}
	return ids
}

func scan(known map[string]device) []device {
	if !loadGC() {
		return nil
	}
	var found []device
	withPool(func() {
		for _, c := range controllers() {
			if c.Send(selExtended) == 0 {
				continue // a remote or a micro gamepad
			}
			k := strconv.FormatUint(uint64(c), 16)
			if d, ok := known[k]; ok {
				found = append(found, d)
				continue
			}
			d := &gcPad{ctl: c.Send(selRetain)}
			if v := c.Send(selVendor); v != 0 {
				d.title = goString(objc.Send[uintptr](v, selUTF8))
			}
			found = append(found, d)
		}
	})
	return found
}

func goString(p uintptr) string {
	if p == 0 {
		return ""
	}
	b := unsafe.Slice((*byte)(*(*unsafe.Pointer)(unsafe.Pointer(&p))), 256)
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}

func (d *gcPad) key() string  { return strconv.FormatUint(uint64(d.ctl), 16) }
func (d *gcPad) name() string { return d.title }
func (d *gcPad) close()       { d.ctl.Send(selRelease) }

// gcPresent reports whether a controller is still among the controllers;
// tests replace it.
var gcPresent = func(ctl objc.ID) bool {
	for _, c := range controllers() {
		if c == ctl {
			return true
		}
	}
	return false
}

func value(input objc.ID) float32 {
	if input == 0 {
		return 0
	}
	return sendFloat(input, selValue)
}

func (d *gcPad) read(s *State) bool {
	ok := false
	withPool(func() {
		if ok = gcPresent(d.ctl); !ok {
			return
		}
		gp := d.ctl.Send(selExtended)
		if gp == 0 {
			ok = false
			return
		}
		for b, sel := range buttonSels {
			// Older systems have no Home or Options button.
			if sel != 0 && objc.Send[bool](gp, selResponds, sel) {
				s.Buttons[b] = value(gp.Send(sel))
			}
		}
		if pad := gp.Send(selDpad); pad != 0 {
			s.Buttons[Up] = value(pad.Send(selUp))
			s.Buttons[Down] = value(pad.Send(selDown))
			s.Buttons[Left] = value(pad.Send(selLeft))
			s.Buttons[Right] = value(pad.Send(selRight))
		}
		// GameController's y axes go up.
		if st := gp.Send(selLeftStick); st != 0 {
			s.Axes[LeftX], s.Axes[LeftY] = value(st.Send(selXAxis)), -value(st.Send(selYAxis))
		}
		if st := gp.Send(selRightStick); st != 0 {
			s.Axes[RightX], s.Axes[RightY] = value(st.Send(selXAxis)), -value(st.Send(selYAxis))
		}
	})
	return ok
}
