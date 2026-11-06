//go:build darwin

#include "textflag.h"

// The callbacks of Core Text's run delegates, which lay out inline
// objects: getters return a CGFloat, which Go callbacks cannot, so they
// are leaves reading it from the object's metrics their refCon points at,
// {ascent, descent, width} in float64s.

TEXT ·runDelegateDealloc(SB), NOSPLIT|NOFRAME, $0-0
	RET

TEXT ·runDelegateAscent(SB), NOSPLIT|NOFRAME, $0-0
	MOVSD 0(DI), X0
	RET

TEXT ·runDelegateDescent(SB), NOSPLIT|NOFRAME, $0-0
	MOVSD 8(DI), X0
	RET

TEXT ·runDelegateWidth(SB), NOSPLIT|NOFRAME, $0-0
	MOVSD 16(DI), X0
	RET

// func runDelegateFuncs() (dealloc, ascent, descent, width uintptr)
TEXT ·runDelegateFuncs(SB), NOSPLIT, $0-32
	LEAQ ·runDelegateDealloc(SB), AX
	MOVQ AX, dealloc+0(FP)
	LEAQ ·runDelegateAscent(SB), AX
	MOVQ AX, ascent+8(FP)
	LEAQ ·runDelegateDescent(SB), AX
	MOVQ AX, descent+16(FP)
	LEAQ ·runDelegateWidth(SB), AX
	MOVQ AX, width+24(FP)
	RET
