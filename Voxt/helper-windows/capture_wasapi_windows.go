//go:build windows

package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"golang.org/x/sys/windows"
)

const captureWatchdogTimeout = 1200 * time.Millisecond

func (session *captureSession) runMicrophone() (string, *proto.Error) {
	devices, err := captureDevices()
	if err != nil {
		return "", deviceUnavailable(firstRequested(session.params.Devices), err.Error())
	}
	candidates := initialDeviceCandidates(devices, session.params.Devices)
	stream, device, failures, lastErr := openFirstCandidate(candidates)
	if stream == nil {
		message := "No input device is available"
		if lastErr != nil {
			message = lastErr.Error()
		}
		return "", deviceUnavailable(firstRequested(session.params.Devices), message)
	}
	if len(session.params.Devices) > 0 && device.ID != session.params.Devices[0] {
		session.fellBack = true
		session.fallbackReason = strings.Join(failures, "; ")
	}
	session.startSource(device)

	for {
		reason := session.pump(stream)
		stream.close()
		switch reason {
		case proto.CaptureStopped, proto.CaptureMaxSeconds, "":
			return restartFailureReason(reason), nil
		}
		if reason == proto.CaptureNoCallback {
			session.silentRestarts++
			if session.silentRestarts >= 3 {
				if session.started {
					return proto.CaptureNoCallback, nil
				}
				return "", deviceUnavailable(session.current.ID, "No input gave audio")
			}
		}
		devices, err = captureDevices()
		if err != nil {
			if session.started {
				return restartFailureReason(reason), nil
			}
			return "", deviceUnavailable(session.current.ID, err.Error())
		}
		candidates = session.restartCandidates(devices, reason)
		stream, device, _, lastErr = openFirstCandidate(candidates)
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
		if stream == nil {
			if session.started {
				return restartFailureReason(reason), nil
			}
			message := "No input gave audio"
			if lastErr != nil {
				message = lastErr.Error()
			}
			return "", deviceUnavailable(session.current.ID, message)
		}
		session.restarts++
		session.startSource(device)
		session.server.notify(proto.MethodCaptureRestarted, proto.RestartedParams{
			Job: session.jobID(), Reason: reason, Attempt: session.restarts,
			Device: session.current.ID, DeviceName: session.current.Name,
		})
	}
}

func (session *captureSession) pump(stream *wasapiStream) string {
	lastPacket := time.Now()
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
				lastPacket = time.Now()
				session.silentRestarts = 0
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
		}
		if time.Since(lastPacket) >= captureWatchdogTimeout {
			return proto.CaptureNoCallback
		}
		if time.Since(lastDeviceCheck) >= 250*time.Millisecond {
			if !stream.active() {
				return proto.CaptureDeviceLost
			}
			if session.current.Default {
				defaultID, err := defaultCaptureDeviceID()
				if err == nil && defaultID != "" && defaultID != session.current.ID {
					return proto.CaptureConfigurationChange
				}
			}
			lastDeviceCheck = time.Now()
		}
	}
}

func initialDeviceCandidates(devices []proto.CaptureDevice, requested []string) []proto.CaptureDevice {
	candidates := make([]proto.CaptureDevice, 0, len(requested)+1)
	seen := make(map[string]bool)
	for _, id := range requested {
		for _, device := range devices {
			if device.ID == id && !seen[id] {
				candidates = append(candidates, device)
				seen[id] = true
			}
		}
	}
	for _, device := range devices {
		if device.Default && !seen[device.ID] {
			candidates = append(candidates, device)
			seen[device.ID] = true
		}
	}
	if len(candidates) == 0 && len(devices) > 0 {
		candidates = append(candidates, devices[0])
	}
	return candidates
}

func (session *captureSession) restartCandidates(devices []proto.CaptureDevice, reason string) []proto.CaptureDevice {
	defaultDevice, hasDefault := defaultCaptureDevice(devices)
	switch reason {
	case proto.CaptureNoCallback:
		if session.silentRestarts == 1 {
			return []proto.CaptureDevice{session.current}
		}
	case proto.CaptureDeviceLost:
		foundCurrent := false
		var candidates []proto.CaptureDevice
		for _, requestedID := range session.params.Devices {
			if requestedID == session.current.ID {
				foundCurrent = true
				continue
			}
			if foundCurrent {
				for _, device := range devices {
					if device.ID == requestedID {
						candidates = append(candidates, device)
					}
				}
			}
		}
		if foundCurrent {
			if hasDefault && defaultDevice.ID != session.current.ID {
				candidates = append(candidates, defaultDevice)
			}
			return candidates
		}
	case proto.CaptureConfigurationChange:
	default:
		return nil
	}
	if hasDefault {
		return []proto.CaptureDevice{defaultDevice}
	}
	return nil
}

func defaultCaptureDevice(devices []proto.CaptureDevice) (proto.CaptureDevice, bool) {
	for _, device := range devices {
		if device.Default {
			return device, true
		}
	}
	return proto.CaptureDevice{}, false
}

func openFirstCandidate(candidates []proto.CaptureDevice) (*wasapiStream, proto.CaptureDevice, []string, error) {
	var failures []string
	var lastErr error
	for _, candidate := range candidates {
		stream, details, err := openWasapiStream(candidate.ID)
		if err != nil {
			failures = append(failures, fmt.Sprintf("input %s: %v", candidate.ID, err))
			lastErr = err
			continue
		}
		details.Default = candidate.Default
		details.Channels = stream.format.channels
		details.SampleRate = stream.format.sampleRate
		return stream, details, failures, nil
	}
	return nil, proto.CaptureDevice{}, failures, lastErr
}

func defaultCaptureDeviceID() (string, error) {
	enumerator, err := newMMDeviceEnumerator()
	if err != nil {
		return "", err
	}
	defer enumerator.release()
	device, err := enumerator.output(mmDeviceDefaultEndpoint, dataFlowCapture, roleConsole)
	if err != nil {
		return "", err
	}
	defer device.release()
	return mmDeviceID(device)
}

func restartFailureReason(reason string) string {
	if reason == proto.CaptureConfigurationChange {
		return proto.CaptureDeviceLost
	}
	return reason
}

func firstRequested(requested []string) string {
	if len(requested) > 0 {
		return requested[0]
	}
	return ""
}
