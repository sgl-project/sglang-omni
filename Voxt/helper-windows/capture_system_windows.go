//go:build windows

package main

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"golang.org/x/sys/windows"
)

func systemCaptureParams(params proto.SystemParams) proto.CaptureParams {
	return proto.CaptureParams{
		SampleRate: params.SampleRate,
		FrameMS:    params.FrameMS,
		LevelHz:    params.LevelHz,
		MaxSeconds: params.MaxSeconds,
	}
}

func (session *captureSession) runSystem() (string, *proto.Error) {
	stream, device, err := openWasapiLoopbackStream()
	if err != nil {
		return "", deviceUnavailable("", err.Error())
	}
	session.startSource(device)
	for {
		reason := session.pumpSystem(stream)
		stream.close()
		switch reason {
		case proto.CaptureStopped, proto.CaptureMaxSeconds, "":
			return restartFailureReason(reason), nil
		}
		stream, device, err = openWasapiLoopbackStream()
		if session.job.stopped() {
			if stream != nil {
				stream.close()
			}
			return proto.CaptureStopped, nil
		}
		if session.job.cancelled() {
			if stream != nil {
				stream.close()
			}
			return "", nil
		}
		if err != nil {
			if session.started {
				return restartFailureReason(reason), nil
			}
			return "", deviceUnavailable("", err.Error())
		}
		session.restarts++
		session.startSource(device)
		session.server.notify(proto.MethodCaptureRestarted, proto.RestartedParams{
			Job: session.jobID(), Reason: reason, Attempt: session.restarts,
			Device: session.current.ID, DeviceName: "",
		})
	}
}

func (session *captureSession) pumpSystem(stream *wasapiStream) string {
	lastDeviceCheck := time.Now()
	for {
		if session.job.stopped() {
			return proto.CaptureStopped
		}
		if session.job.cancelled() {
			return ""
		}
		waitResult, _ := windows.WaitForSingleObject(stream.event, 50)
		if waitResult == windows.WAIT_OBJECT_0 {
			for {
				samples, err := stream.readPacket()
				if err != nil {
					return proto.CaptureDeviceLost
				}
				if len(samples) == 0 {
					break
				}
				session.accept(samples)
				if session.maxSecondsReached() {
					return proto.CaptureMaxSeconds
				}
				if session.job.stopped() {
					return proto.CaptureStopped
				}
				if session.job.cancelled() {
					return ""
				}
			}
		} else {
			session.accept(make([]float64, max(1, stream.format.sampleRate/20)))
			if session.maxSecondsReached() {
				return proto.CaptureMaxSeconds
			}
		}
		if time.Since(lastDeviceCheck) >= 250*time.Millisecond {
			if !stream.active() {
				return proto.CaptureDeviceLost
			}
			defaultID, err := defaultRenderDeviceID()
			if err != nil {
				return proto.CaptureDeviceLost
			}
			if defaultID != session.current.ID {
				return proto.CaptureConfigurationChange
			}
			lastDeviceCheck = time.Now()
		}
	}
}

func defaultRenderDeviceID() (string, error) {
	enumerator, err := newMMDeviceEnumerator()
	if err != nil {
		return "", err
	}
	defer enumerator.release()
	device, err := enumerator.output(mmDeviceDefaultEndpoint, dataFlowRender, roleConsole)
	if err != nil {
		return "", err
	}
	defer device.release()
	return mmDeviceID(device)
}
