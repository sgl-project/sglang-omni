//go:build windows

package main

import (
	"fmt"
	"maps"
	"reflect"
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

type modifierMask uint16

const (
	modifierLeftShift modifierMask = 1 << iota
	modifierRightShift
	modifierLeftControl
	modifierRightControl
	modifierLeftOption
	modifierRightOption
	modifierLeftCommand
	modifierRightCommand
	modifierFn
)

type hotkeyPress struct {
	virtualKey uint16
	button     int
	mouse      bool
}

type recognizedBinding struct {
	proto.HotkeyBinding
	virtualKey uint16
}

type recognizerConfig struct {
	bindings    []recognizedBinding
	chordOrder  []int
	keyOrder    []int
	doubleTapMS float64
	longPressMS float64
	tapMaxMS    float64
	swallowFn   bool
	active      bool
	cancel      *recognizedBinding
}

type hotkeyFired struct {
	binding  string
	phase    string
	behavior string
	heldMS   float64
	timeMS   float64
}

type firedDeadline struct {
	binding *recognizedBinding
	timeMS  float64
}

type pendingChordTap struct {
	binding *recognizedBinding
	timeMS  float64
	heldMS  float64
}

type doubleChordArm struct {
	chord   modifierMask
	untilMS float64
}

type doubleKeyArm struct {
	press   hotkeyPress
	untilMS float64
}

type hotkeyRecognizer struct {
	config        recognizerConfig
	pressed       modifierMask
	chordStartMS  float64
	chordActive   bool
	chordMax      modifierMask
	spoiled       bool
	longFired     *recognizedBinding
	longDeadline  *firedDeadline
	doubleArm     *doubleChordArm
	pendingTap    *pendingChordTap
	swallowedKeys map[hotkeyPress]bool
	keyDownAt     map[hotkeyPress]float64
	keyLongFired  map[hotkeyPress]*recognizedBinding
	keyDoubleArm  *doubleKeyArm
	missedEvents  int
}

type recognizerResult struct {
	events  []hotkeyFired
	swallow bool
}

func newHotkeyRecognizer() *hotkeyRecognizer {
	return &hotkeyRecognizer{
		swallowedKeys: make(map[hotkeyPress]bool),
		keyDownAt:     make(map[hotkeyPress]float64),
		keyLongFired:  make(map[hotkeyPress]*recognizedBinding),
	}
}

func makeRecognizerConfig(params proto.HotkeyConfigureParams, doubleClickMS float64) (recognizerConfig, error) {
	config := recognizerConfig{
		doubleTapMS: params.DoubleTapMS,
		longPressMS: params.LongPressMS,
		tapMaxMS:    params.TapMaxMS,
		swallowFn:   params.SwallowFn,
		active:      params.Active,
	}
	if config.doubleTapMS == 0 {
		config.doubleTapMS = doubleClickMS
	}
	if config.longPressMS == 0 {
		config.longPressMS = 300
	}
	if config.tapMaxMS == 0 {
		config.tapMaxMS = 1500
	}
	if config.doubleTapMS < 0 || config.longPressMS < 0 || config.tapMaxMS < 0 {
		return config, fmt.Errorf("hotkey timings cannot be negative")
	}
	for _, binding := range params.Bindings {
		recognized, err := recognizeBinding(binding)
		if err != nil {
			return config, err
		}
		config.bindings = append(config.bindings, recognized)
	}
	if params.Cancel != nil {
		cancel, err := recognizeBinding(*params.Cancel)
		if err != nil {
			return config, fmt.Errorf("cancel binding: %w", err)
		}
		cancel.ID = proto.HotkeyCancel
		cancel.Behavior = proto.BehaviorTap
		config.cancel = &cancel
	}
	config.reorder()
	return config, nil
}

func recognizeBinding(binding proto.HotkeyBinding) (recognizedBinding, error) {
	recognized := recognizedBinding{HotkeyBinding: binding}
	if binding.ID == "" || binding.Key != nil && binding.Button != nil || binding.Key == nil && binding.Button == nil && len(binding.Modifiers) == 0 {
		return recognized, fmt.Errorf("invalid binding %q", binding.ID)
	}
	if !slices.Contains([]string{proto.BehaviorTap, proto.BehaviorDoubleTap, proto.BehaviorLongPress}, binding.Behavior) {
		return recognized, fmt.Errorf("binding %q has unknown behavior %q", binding.ID, binding.Behavior)
	}
	if binding.Button != nil && *binding.Button < 2 {
		return recognized, fmt.Errorf("binding %q cannot use button %d", binding.ID, *binding.Button)
	}
	for family, side := range binding.Modifiers {
		if !slices.Contains([]string{proto.ModFn, proto.ModShift, proto.ModControl, proto.ModOption, proto.ModCommand}, family) ||
			!slices.Contains([]string{proto.SideAny, proto.SideLeft, proto.SideRight}, side) || family == proto.ModFn && side != proto.SideAny {
			return recognized, fmt.Errorf("binding %q has invalid modifier %s=%s", binding.ID, family, side)
		}
	}
	if binding.Key != nil {
		virtualKey, err := windowsVirtualKey(*binding.Key)
		if err != nil {
			return recognized, fmt.Errorf("binding %q: %w", binding.ID, err)
		}
		recognized.virtualKey = virtualKey
	}
	return recognized, nil
}

func (config *recognizerConfig) reorder() {
	for index := range config.bindings {
		if _, isPress := config.bindings[index].press(); isPress {
			config.keyOrder = append(config.keyOrder, index)
		} else {
			config.chordOrder = append(config.chordOrder, index)
		}
	}
	specificity := func(index int) int {
		binding := config.bindings[index]
		score := 2 * len(binding.Modifiers)
		for _, side := range binding.Modifiers {
			if side != proto.SideAny {
				score++
			}
		}
		return score
	}
	slices.SortStableFunc(config.chordOrder, func(first, second int) int { return specificity(second) - specificity(first) })
	slices.SortStableFunc(config.keyOrder, func(first, second int) int { return specificity(second) - specificity(first) })
}

func (binding *recognizedBinding) press() (hotkeyPress, bool) {
	if binding.Key != nil {
		return hotkeyPress{virtualKey: binding.virtualKey}, true
	}
	if binding.Button != nil {
		return hotkeyPress{button: *binding.Button, mouse: true}, true
	}
	return hotkeyPress{}, false
}

func modifierFamilyMask(family string) modifierMask {
	switch family {
	case proto.ModShift:
		return modifierLeftShift | modifierRightShift
	case proto.ModControl:
		return modifierLeftControl | modifierRightControl
	case proto.ModOption:
		return modifierLeftOption | modifierRightOption
	case proto.ModCommand:
		return modifierLeftCommand | modifierRightCommand
	case proto.ModFn:
		return modifierFn
	default:
		return 0
	}
}

func bindingMatchesChord(binding *recognizedBinding, chord modifierMask) bool {
	for _, family := range []string{proto.ModFn, proto.ModShift, proto.ModControl, proto.ModOption, proto.ModCommand} {
		pressed := chord & modifierFamilyMask(family)
		side, required := binding.Modifiers[family]
		if !required {
			if pressed != 0 {
				return false
			}
			continue
		}
		if pressed == 0 {
			return false
		}
		switch family {
		case proto.ModShift:
			if side == proto.SideLeft && pressed&modifierRightShift != 0 || side == proto.SideRight && pressed&modifierLeftShift != 0 {
				return false
			}
		case proto.ModControl:
			if side == proto.SideLeft && pressed&modifierRightControl != 0 || side == proto.SideRight && pressed&modifierLeftControl != 0 {
				return false
			}
		case proto.ModOption:
			if side == proto.SideLeft && pressed&modifierRightOption != 0 || side == proto.SideRight && pressed&modifierLeftOption != 0 {
				return false
			}
		case proto.ModCommand:
			if side == proto.SideLeft && pressed&modifierRightCommand != 0 || side == proto.SideRight && pressed&modifierLeftCommand != 0 {
				return false
			}
		}
	}
	return true
}

func bindingMatchesPress(binding *recognizedBinding, press hotkeyPress, modifiers modifierMask) bool {
	bindingPress, ok := binding.press()
	if !ok || bindingPress != press {
		return false
	}
	if !press.mouse && fnFlaggedWindowsKey(press.virtualKey) {
		copy := *binding
		copy.Modifiers = maps.Clone(binding.Modifiers)
		delete(copy.Modifiers, proto.ModFn)
		return bindingMatchesChord(&copy, modifiers&^modifierFn)
	}
	return bindingMatchesChord(binding, modifiers)
}

func fnFlaggedWindowsKey(virtualKey uint16) bool {
	return virtualKey >= 0x70 && virtualKey <= 0x83 || virtualKey >= 0x24 && virtualKey <= 0x28 ||
		virtualKey == 0x21 || virtualKey == 0x22 || virtualKey == 0x2e || virtualKey == 0x2f
}

func modifierForVirtualKey(virtualKey uint16) (modifierMask, bool) {
	switch virtualKey {
	case 0x10, 0xa0:
		return modifierLeftShift, true
	case 0xa1:
		return modifierRightShift, true
	case 0x11, 0xa2:
		return modifierLeftControl, true
	case 0xa3:
		return modifierRightControl, true
	case 0x12, 0xa4:
		return modifierLeftOption, true
	case 0xa5:
		return modifierRightOption, true
	case 0x5b:
		return modifierLeftCommand, true
	case 0x5c:
		return modifierRightCommand, true
	default:
		return 0, false
	}
}

func (recognizer *hotkeyRecognizer) configure(config recognizerConfig) {
	sameBindings := reflect.DeepEqual(config.bindings, recognizer.config.bindings) && reflect.DeepEqual(config.cancel, recognizer.config.cancel)
	recognizer.config = config
	if !sameBindings {
		recognizer.reset()
	}
}

func (recognizer *hotkeyRecognizer) reset() {
	recognizer.resetChord()
	recognizer.doubleArm = nil
	recognizer.pendingTap = nil
	recognizer.keyDoubleArm = nil
	recognizer.keyLongFired = make(map[hotkeyPress]*recognizedBinding)
	recognizer.keyDownAt = make(map[hotkeyPress]float64)
}

func (recognizer *hotkeyRecognizer) resetChord() {
	recognizer.chordActive = false
	recognizer.chordMax = 0
	recognizer.spoiled = false
	recognizer.longFired = nil
	recognizer.longDeadline = nil
}

func (recognizer *hotkeyRecognizer) nextDeadline() (float64, bool) {
	var deadline float64
	found := false
	if !recognizer.chordActive && recognizer.pendingTap != nil {
		deadline, found = recognizer.pendingTap.timeMS, true
	}
	if recognizer.longDeadline != nil && (!found || recognizer.longDeadline.timeMS < deadline) {
		deadline, found = recognizer.longDeadline.timeMS, true
	}
	return deadline, found
}

func (recognizer *hotkeyRecognizer) tick(nowMS float64) []hotkeyFired {
	var events []hotkeyFired
	if !recognizer.chordActive && recognizer.pendingTap != nil && nowMS >= recognizer.pendingTap.timeMS {
		pending := recognizer.pendingTap
		events = append(events, fire(pending.binding, proto.PhaseTrigger, pending.heldMS, pending.timeMS))
		recognizer.pendingTap = nil
		recognizer.doubleArm = nil
	}
	if recognizer.longDeadline != nil && nowMS >= recognizer.longDeadline.timeMS {
		deadline := recognizer.longDeadline
		recognizer.longDeadline = nil
		if !recognizer.spoiled && recognizer.pressed == recognizer.chordMax && bindingMatchesChord(deadline.binding, recognizer.pressed) {
			recognizer.longFired = deadline.binding
			events = append(events, fireWithBehavior(deadline.binding, proto.PhaseDown, deadline.timeMS-recognizer.chordStartMS, deadline.timeMS))
		}
	}
	return events
}

func (recognizer *hotkeyRecognizer) modifier(virtualKey uint16, now modifierMask, timeMS float64) recognizerResult {
	events := recognizer.tick(timeMS)
	modifier, _ := modifierForVirtualKey(virtualKey)
	was := recognizer.pressed
	if (was&modifier != 0) == (now&modifier != 0) {
		recognizer.missedEvents++
		recognizer.resetChord()
		recognizer.doubleArm = nil
		recognizer.pendingTap = nil
		if now&modifier != 0 {
			was = now &^ modifier
		} else {
			was = now
		}
	}
	recognizer.pressed = now
	if was == 0 && now != 0 {
		recognizer.chordStartMS = timeMS
		recognizer.chordActive = true
		recognizer.chordMax = now
		recognizer.spoiled = false
		recognizer.longFired = nil
		if recognizer.doubleArm != nil && timeMS > recognizer.doubleArm.untilMS {
			recognizer.doubleArm = nil
		}
	} else {
		recognizer.chordMax |= now
	}
	if now == 0 && was != 0 {
		events = append(events, recognizer.releaseChord(timeMS)...)
	} else if now != 0 {
		recognizer.longDeadline = nil
		if !recognizer.spoiled && now == recognizer.chordMax && recognizer.longFired == nil {
			for _, index := range recognizer.config.chordOrder {
				binding := &recognizer.config.bindings[index]
				if binding.Behavior == proto.BehaviorLongPress && bindingMatchesChord(binding, now) {
					recognizer.longDeadline = &firedDeadline{binding: binding, timeMS: timeMS + recognizer.config.longPressMS}
					break
				}
			}
		}
	}
	return recognizerResult{events: events}
}

func (recognizer *hotkeyRecognizer) releaseChord(timeMS float64) []hotkeyFired {
	defer recognizer.resetChord()
	heldMS := timeMS - recognizer.chordStartMS
	if recognizer.longFired != nil {
		recognizer.doubleArm = nil
		return []hotkeyFired{fireWithBehavior(recognizer.longFired, proto.PhaseUp, heldMS, timeMS)}
	}
	if recognizer.spoiled || heldMS > recognizer.config.tapMaxMS {
		recognizer.doubleArm = nil
		return nil
	}
	chord := recognizer.chordMax
	if recognizer.config.active && recognizer.config.cancel != nil {
		if _, isPress := recognizer.config.cancel.press(); !isPress && bindingMatchesChord(recognizer.config.cancel, chord) {
			recognizer.pendingTap = nil
			recognizer.doubleArm = nil
			return []hotkeyFired{fire(recognizer.config.cancel, proto.PhaseTrigger, heldMS, timeMS)}
		}
	}
	var tapBinding, doubleBinding *recognizedBinding
	for _, index := range recognizer.config.chordOrder {
		binding := &recognizer.config.bindings[index]
		if !bindingMatchesChord(binding, chord) {
			continue
		}
		if tapBinding == nil && binding.Behavior == proto.BehaviorTap {
			tapBinding = binding
		}
		if doubleBinding == nil && binding.Behavior == proto.BehaviorDoubleTap {
			doubleBinding = binding
		}
	}
	if recognizer.config.active && (tapBinding != nil || doubleBinding != nil) {
		binding := tapBinding
		if binding == nil {
			binding = doubleBinding
		}
		recognizer.pendingTap = nil
		recognizer.doubleArm = nil
		return []hotkeyFired{fire(binding, proto.PhaseTrigger, heldMS, timeMS)}
	}
	if recognizer.doubleArm != nil && recognizer.doubleArm.chord == chord && recognizer.chordStartMS <= recognizer.doubleArm.untilMS && doubleBinding != nil {
		recognizer.doubleArm = nil
		recognizer.pendingTap = nil
		return []hotkeyFired{fire(doubleBinding, proto.PhaseTrigger, heldMS, timeMS)}
	}
	var events []hotkeyFired
	if recognizer.pendingTap != nil {
		events = append(events, fire(recognizer.pendingTap.binding, proto.PhaseTrigger, recognizer.pendingTap.heldMS, timeMS))
		recognizer.pendingTap = nil
	}
	recognizer.doubleArm = nil
	if doubleBinding != nil {
		recognizer.doubleArm = &doubleChordArm{chord: chord, untilMS: timeMS + recognizer.config.doubleTapMS}
		if tapBinding != nil {
			recognizer.pendingTap = &pendingChordTap{binding: tapBinding, timeMS: timeMS + recognizer.config.doubleTapMS, heldMS: heldMS}
		}
	} else if tapBinding != nil {
		events = append(events, fire(tapBinding, proto.PhaseTrigger, heldMS, timeMS))
	}
	return events
}

func (recognizer *hotkeyRecognizer) keyDown(press hotkeyPress, repeat bool, timeMS float64) recognizerResult {
	events := recognizer.tick(timeMS)
	if recognizer.pressed != 0 {
		recognizer.spoiled = true
		recognizer.longDeadline = nil
	}
	if recognizer.config.active && recognizer.config.cancel != nil && bindingMatchesPress(recognizer.config.cancel, press, recognizer.pressed) {
		if !repeat {
			events = append(events, fire(recognizer.config.cancel, proto.PhaseTrigger, 0, timeMS))
		}
		recognizer.swallowedKeys[press] = true
		return recognizerResult{events: events, swallow: true}
	}
	var matched *recognizedBinding
	for _, index := range recognizer.config.keyOrder {
		binding := &recognizer.config.bindings[index]
		if bindingMatchesPress(binding, press, recognizer.pressed) {
			matched = binding
			break
		}
	}
	if matched == nil {
		return recognizerResult{events: events, swallow: recognizer.swallowedKeys[press]}
	}
	swallow := matched.Swallow == nil || *matched.Swallow
	if swallow {
		recognizer.swallowedKeys[press] = true
	}
	if repeat {
		return recognizerResult{events: events, swallow: swallow}
	}
	recognizer.keyDownAt[press] = timeMS
	switch matched.Behavior {
	case proto.BehaviorTap:
		events = append(events, fire(matched, proto.PhaseTrigger, 0, timeMS))
	case proto.BehaviorLongPress:
		recognizer.keyLongFired[press] = matched
		events = append(events, fire(matched, proto.PhaseDown, 0, timeMS))
	case proto.BehaviorDoubleTap:
		if recognizer.keyDoubleArm != nil && recognizer.keyDoubleArm.press == press && timeMS <= recognizer.keyDoubleArm.untilMS {
			recognizer.keyDoubleArm = nil
			events = append(events, fire(matched, proto.PhaseTrigger, 0, timeMS))
		} else {
			recognizer.keyDoubleArm = &doubleKeyArm{press: press, untilMS: timeMS + recognizer.config.doubleTapMS}
		}
	}
	return recognizerResult{events: events, swallow: swallow}
}

func (recognizer *hotkeyRecognizer) keyUp(press hotkeyPress, timeMS float64) recognizerResult {
	events := recognizer.tick(timeMS)
	if binding := recognizer.keyLongFired[press]; binding != nil {
		events = append(events, fireWithBehavior(binding, proto.PhaseUp, timeMS-recognizer.keyDownAt[press], timeMS))
		delete(recognizer.keyLongFired, press)
	}
	delete(recognizer.keyDownAt, press)
	swallow := recognizer.swallowedKeys[press]
	delete(recognizer.swallowedKeys, press)
	return recognizerResult{events: events, swallow: swallow}
}

func fire(binding *recognizedBinding, phase string, heldMS, timeMS float64) hotkeyFired {
	return fireWithBehavior(binding, phase, heldMS, timeMS)
}

func fireWithBehavior(binding *recognizedBinding, phase string, heldMS, timeMS float64) hotkeyFired {
	return hotkeyFired{binding: binding.ID, phase: phase, behavior: binding.Behavior, heldMS: heldMS, timeMS: timeMS}
}

func (config recognizerConfig) hasModifierChords() bool {
	for _, binding := range config.bindings {
		if _, isPress := binding.press(); !isPress {
			return true
		}
	}
	if config.cancel != nil {
		if _, isPress := config.cancel.press(); !isPress {
			return true
		}
	}
	return false
}
