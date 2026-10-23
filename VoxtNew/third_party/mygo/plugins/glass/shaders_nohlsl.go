//go:build !windows

package glass

// The Direct3D bytecode compiled ahead of time is for Windows alone.
var hlslBytecode, blurBytecode []byte

const hlslSum, blurHLSLSum = "", ""
