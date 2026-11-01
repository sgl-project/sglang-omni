// Command fakehelper is helpertest's fake voxtnew-helper as a program,
// to try the app without the Swift helper (on Linux, or on a Mac without
// Xcode): build it and point VOXTNEW_HELPER at it,
//
//	go build -o /tmp/fakehelper ./internal/helper/helpertest/cmd/fakehelper
//	VOXTNEW_HELPER=/tmp/fakehelper go tool mygo dev
//
// It takes the options of VOXTNEW_FAKE_HELPER (helpertest) from its
// arguments or that variable; it records a tone, not the microphone.
package main

import (
	"os"
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
)

func main() {
	opts := os.Getenv("VOXTNEW_FAKE_HELPER")
	if len(os.Args) > 1 {
		opts = strings.Join(os.Args[1:], ",")
	}
	helpertest.Run(opts)
}
