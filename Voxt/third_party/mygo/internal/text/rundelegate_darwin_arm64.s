//go:build darwin

#include "textflag.h"

// The callbacks of Core Text's run delegates, which lay out inline
// objects: getters return a CGFloat, which Go callbacks cannot, so they
// are leaves reading it from the object's metrics their refCon points at,
// {ascent, descent, width} in float64s.

TEXT ·runDelegateDealloc(SB), NOSPLIT|NOFRAME, $0-0
	RET

TEXT ·runDelegateAscent(SB), NOSPLIT|NOFRAME, $0-0
	FMOVD 0(R0), F0
	RET

TEXT ·runDelegateDescent(SB), NOSPLIT|NOFRAME, $0-0
	FMOVD 8(R0), F0
	RET

TEXT ·runDelegateWidth(SB), NOSPLIT|NOFRAME, $0-0
	FMOVD 16(R0), F0
	RET

// func runDelegateFuncs() (dealloc, ascent, descent, width uintptr)
TEXT ·runDelegateFuncs(SB), NOSPLIT, $0-32
	MOVD $·runDelegateDealloc(SB), R0
	MOVD R0, dealloc+0(FP)
	MOVD $·runDelegateAscent(SB), R0
	MOVD R0, ascent+8(FP)
	MOVD $·runDelegateDescent(SB), R0
	MOVD R0, descent+16(FP)
	MOVD $·runDelegateWidth(SB), R0
	MOVD R0, width+24(FP)
	RET
