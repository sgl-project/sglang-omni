//go:build windows

package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

func TestCaptureSystemJobCoexistsWithMicrophone(t *testing.T) {
	server := &helperServer{jobs: make(map[string]*helperJob)}
	if _, jobErr := server.registerJob("microphone", proto.MethodCaptureStart); jobErr != nil {
		t.Fatal(jobErr)
	}
	if _, jobErr := server.registerJob("system", proto.MethodCaptureSystem); jobErr != nil {
		t.Fatalf("system capture could not run beside microphone capture: %v", jobErr)
	}
	if _, jobErr := server.registerJob("second-system", proto.MethodCaptureSystem); jobErr == nil || jobErr.Code != proto.CodeBusy {
		t.Fatalf("second system capture = %v, want busy", jobErr)
	}
	if _, jobErr := server.registerJob("second-microphone", proto.MethodCaptureStart); jobErr == nil || jobErr.Code != proto.CodeBusy {
		t.Fatalf("second microphone capture = %v, want busy", jobErr)
	}
}

func TestSystemCaptureParams(t *testing.T) {
	params := systemCaptureParams(proto.SystemParams{SampleRate: 48000, FrameMS: 20, LevelHz: 10, MaxSeconds: 2.5})
	if params.SampleRate != 48000 || params.FrameMS != 20 || params.LevelHz != 10 || params.MaxSeconds != 2.5 {
		t.Fatalf("converted params = %+v", params)
	}
	if len(params.Devices) != 0 {
		t.Fatalf("system capture inherited input choices: %v", params.Devices)
	}
}
