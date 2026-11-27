//go:build linux && (amd64 || arm64)

package linux

import (
	"strconv"
	"sync"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

// Hooks for tests of native UI with input methods, file drops and
// assistive technology.

var (
	accessTestsOnce               sync.Once
	gSignalEmitString             func(obj ptr, signal *byte, s *byte)
	gtkIMMulticontextGetContextID func(im ptr) ptr
	gtkIMMulticontextSetContextID func(im ptr, id *byte)
	atkObjectGetNAccessibleChild  func(obj ptr) int32
	atkObjectRefAccessibleChild   func(obj ptr, i int32) ptr
	atkObjectGetRole              func(obj ptr) int32
	atkRoleGetName                func(role int32) ptr
	atkObjectGetName              func(obj ptr) ptr
	atkObjectRefStateSet          func(obj ptr) ptr
	atkStateSetContainsState      func(set ptr, state int32) bool
	atkActionGetNActions          func(obj ptr) int32
	atkActionDoAction             func(obj ptr, i int32) bool
	atkTextGetText                func(obj ptr, start, end int32) ptr
	atkValueGetValueAndText       func(obj ptr, value *float64, text ptr)
	atkValueSetValue              func(obj ptr, value float64)
	atkEditableTextSetTextContent func(obj ptr, text *byte)
	atkComponentGrabFocus         func(obj ptr) bool
)

func surfaceByHandle(handle uintptr) *surface {
	if w := windowByHandle(handle); w != nil {
		return w.surface
	}
	return nil
}

// TestSurrounding returns the text around the caret that input methods get
// from a window showing native UI, and the caret's offset in runes. GTK's
// own input method reads it back: others, as fcitx5's, keep it to
// themselves, so the window's input switches to GTK's.
func TestSurrounding(handle uintptr) (text string, caret int, ok bool) {
	s := surfaceByHandle(handle)
	if s == nil {
		return "", 0, false
	}
	if gtkIMMulticontextSetContextID == nil {
		mustBind(libGTK, &gtkIMMulticontextGetContextID, "gtk_im_multicontext_get_context_id")
		mustBind(libGTK, &gtkIMMulticontextSetContextID, "gtk_im_multicontext_set_context_id")
	}
	const simple = "gtk-im-context-simple"
	if goStr(gtkIMMulticontextGetContextID(s.im)) != simple {
		gtkIMMulticontextSetContextID(s.im, cs(simple))
	}
	var p ptr
	var cursor int32
	if !gtkIMContextGetSurrounding(s.im, &p, &cursor) {
		return "", 0, true
	}
	text = takeStr(p)
	return text, utf8.RuneCountInString(text[:max(0, min(int(cursor), len(text)))]), true
}

// TestComposeOver does what an input method that edits typed text does:
// it deletes length runes from offset from of the text around the caret
// unless from is -1, then composes text with the caret at caret, or
// commits it.
func TestComposeOver(handle uintptr, text string, caret int, commit bool, from, length int) bool {
	s := surfaceByHandle(handle)
	if s == nil {
		return false
	}
	if from >= 0 {
		_, cursor, _ := TestSurrounding(handle)
		gtkIMContextDeleteSurrounding(s.im, int32(from-cursor), int32(length))
	}
	if commit {
		if gSignalEmitString == nil {
			mustBind(libGObject, &gSignalEmitString, "g_signal_emit_by_name")
		}
		gSignalEmitString(s.im, cs("commit"), cs(text))
	} else {
		// Input methods compose in contexts out of the test's reach: send
		// what the preedit-changed signal would.
		s.send(platform.SurfaceEvent{Kind: platform.TextComposition, Text: text, Caret: caret})
	}
	return true
}

// TestDropFiles drags files to (x, y) of a window showing native UI and
// drops them there, as GTK tells it, and reports whether the content took
// them over there and when dropped.
func TestDropFiles(handle uintptr, x, y float64, paths []string) (over, dropped bool) {
	s := surfaceByHandle(handle)
	if s == nil {
		return false, false
	}
	over = s.fileDragOver(x, y)
	s.send(platform.SurfaceEvent{Kind: platform.FileDragLeave})
	return over, s.dropFiles(x, y, paths)
}

// TestMoveSurfacePointer sends real XTEST motion to the surface, including
// during GTK's native drag tracker. It is unavailable outside X11.
func TestMoveSurfacePointer(handle uintptr, x, y float64) bool {
	s := surfaceByHandle(handle)
	if s == nil {
		return false
	}
	var origin func(ptr, *int32, *int32) int32
	if !bind(libGDK, &origin, "gdk_window_get_origin") {
		return false
	}
	var ox, oy int32
	origin(s.eventWindow(), &ox, &oy)
	scale := float64(gtkWidgetGetScaleFactor(s.area))
	return TestMovePointer(int((float64(ox)+x)*scale), int((float64(oy)+y)*scale))
}

func TestDataDragReady(handle uintptr) bool {
	s := surfaceByHandle(handle)
	return s != nil && s.lastPointer != 0
}

// TestStartSerializedDrag simulates another application: the source uses
// GTK's production providers with no token in MyGo's local registry.
func TestStartSerializedDrag(handle uintptr, d transfer.Data, ops transfer.Operation, done func(transfer.Result)) bool {
	s := surfaceByHandle(handle)
	if s == nil || s.lastPointer == 0 {
		return false
	}
	s.StartDataDrag(platform.DragRequest{Data: d.Snapshot(), Operations: ops, X: 30, Y: 30, Done: done})
	return true
}
func TestCancelDataDrag(handle uintptr) {
	if s := surfaceByHandle(handle); s != nil {
		s.CancelDataDrag()
	}
}

func TestDataDragResources() int { return len(gtkDataSources) }

// TestAccessNode is an element of native UI as assistive technology reads
// it through ATK: the name of its role, its name, and its value.
type TestAccessNode struct {
	Role, Label, Value string
	obj                ptr
}

func loadAccessTests() {
	libATK, _ := open("libatk-1.0.so.0")
	a := libATK
	mustBind(a, &atkObjectGetNAccessibleChild, "atk_object_get_n_accessible_children")
	mustBind(a, &atkObjectRefAccessibleChild, "atk_object_ref_accessible_child")
	mustBind(a, &atkObjectGetRole, "atk_object_get_role")
	mustBind(a, &atkRoleGetName, "atk_role_get_name")
	mustBind(a, &atkObjectGetName, "atk_object_get_name")
	mustBind(a, &atkObjectRefStateSet, "atk_object_ref_state_set")
	mustBind(a, &atkStateSetContainsState, "atk_state_set_contains_state")
	mustBind(a, &atkActionGetNActions, "atk_action_get_n_actions")
	mustBind(a, &atkActionDoAction, "atk_action_do_action")
	mustBind(a, &atkTextGetText, "atk_text_get_text")
	mustBind(a, &atkValueGetValueAndText, "atk_value_get_value_and_text")
	mustBind(a, &atkValueSetValue, "atk_value_set_value")
	mustBind(a, &atkEditableTextSetTextContent, "atk_editable_text_set_text_contents")
	mustBind(a, &atkComponentGrabFocus, "atk_component_grab_focus")
}

// TestAccessibility reads the elements of a window showing native UI as
// assistive technology does, in order, and reports false when it cannot.
func TestAccessibility(handle uintptr) ([]TestAccessNode, bool) {
	s := surfaceByHandle(handle)
	if s == nil || areaType == 0 {
		return nil, false
	}
	accessTestsOnce.Do(loadAccessTests)
	var nodes []TestAccessNode
	var walk func(obj ptr)
	walk = func(obj ptr) {
		for i := range atkObjectGetNAccessibleChild(obj) {
			child := atkObjectRefAccessibleChild(obj, i)
			if child == 0 {
				continue
			}
			n := TestAccessNode{Role: goStr(atkRoleGetName(atkObjectGetRole(child))), Label: goStr(atkObjectGetName(child)), obj: child}
			if n.Role == "push button" {
				n.Role = "button" // as older versions of ATK name it
			}
			states := atkObjectRefStateSet(child)
			switch {
			case gTypeCheckInstanceIsA(child, atkTextGetType()):
				n.Value = takeStr(atkTextGetText(child, 0, -1))
			case gTypeCheckInstanceIsA(child, atkValueGetType()):
				var v float64
				atkValueGetValueAndText(child, &v, 0)
				n.Value = strconv.FormatFloat(v, 'g', -1, 64)
			case n.Role == "check box" || n.Role == "radio button":
				n.Value = "0"
				if atkStateSetContainsState(states, atkStates.checked) {
					n.Value = "1"
				}
			}
			gObjectUnref(states)
			nodes = append(nodes, n)
			walk(child)
			gObjectUnref(child) // the tree holds the node
		}
	}
	walk(gtkWidgetGetAccessible(s.area))
	return nodes, true
}

// TestAccessibilityPerform acts on the element labeled label as assistive
// technology does: "press", "increment", "decrement", "focus", or "value"
// to set its text to value. It reports whether the element can.
func TestAccessibilityPerform(handle uintptr, label, action, value string) bool {
	nodes, _ := TestAccessibility(handle)
	for _, n := range nodes {
		if n.Label != label {
			continue
		}
		switch action {
		case "press":
			return gTypeCheckInstanceIsA(n.obj, atkActionGetType()) && atkActionGetNActions(n.obj) > 0 && atkActionDoAction(n.obj, 0)
		case "increment", "decrement":
			if !gTypeCheckInstanceIsA(n.obj, atkValueGetType()) {
				return false
			}
			var v float64
			atkValueGetValueAndText(n.obj, &v, 0)
			step := 1.0
			if action == "decrement" {
				step = -1
			}
			atkValueSetValue(n.obj, v+step)
			return true
		case "focus":
			return atkComponentGrabFocus(n.obj)
		case "value":
			if !gTypeCheckInstanceIsA(n.obj, atkEditableTextGetType()) {
				return false
			}
			atkEditableTextSetTextContent(n.obj, cs(value))
			return true
		}
		return false
	}
	return false
}
