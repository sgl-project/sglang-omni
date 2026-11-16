//go:build windows

#include "textflag.h"

// UI Automation passes doubles in floating point registers, which Go
// callbacks do not read: these thunks move their bits, and the arguments
// after them, to the integer registers of the arguments' positions, then
// go on to the callbacks. They use only registers a call may change: the
// assembler's loads of globals use R27, which the caller keeps.

// IRawElementProviderFragmentRoot::ElementProviderFromPoint(this, x, y, out)
TEXT ·uiaFromPointThunk(SB), NOSPLIT|NOFRAME, $0-0
	MOVD  R1, R3
	FMOVD F0, R1
	FMOVD F1, R2
	MOVD  $·uiaFromPointCallback(SB), R9
	MOVD  (R9), R9
	B     (R9)

// IRangeValueProvider::SetValue(this, value)
TEXT ·uiaSetValueThunk(SB), NOSPLIT|NOFRAME, $0-0
	FMOVD F0, R1
	MOVD  $·uiaSetValueCallback(SB), R9
	MOVD  (R9), R9
	B     (R9)

// func uiaThunks() (fromPoint, setValue uintptr)
TEXT ·uiaThunks(SB), NOSPLIT, $0-16
	MOVD $·uiaFromPointThunk(SB), R0
	MOVD R0, fromPoint+0(FP)
	MOVD $·uiaSetValueThunk(SB), R0
	MOVD R0, setValue+8(FP)
	RET
