//go:build windows

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"golang.org/x/sys/windows"
)

type captureSession struct {
	server         *helperServer
	job            *helperJob
	id             string
	params         proto.CaptureParams
	startedAt      time.Time
	rate           int
	frameSamples   int
	levelInterval  time.Duration
	current        proto.CaptureDevice
	fellBack       bool
	fallbackReason string
	started        bool
	restarts       int
	silentRestarts int
	peak           float64
	firstFrameMS   float64
	lastLevel      time.Time
	resampler      linearResampler
	framePending   []float64
	sequence       int
	totalSamples   int
}

func (server *helperServer) serveCaptureDevices(message proto.Message) {
	if err := windows.CoInitializeEx(0, windows.COINIT_MULTITHREADED); err != nil {
		server.fail(message.ID, proto.CodeInternal, "Could not initialize WASAPI: %v", err)
		return
	}
	defer windows.CoUninitialize()
	devices, err := captureDevices()
	if err != nil {
		server.fail(message.ID, proto.CodeInternal, "Could not enumerate audio inputs: %v", err)
		return
	}
	server.result(message.ID, proto.DevicesResult{Devices: devices})
}

func (server *helperServer) startCapture(message proto.Message) {
	var params proto.CaptureParams
	if err := message.DecodeParams(&params); err != nil {
		server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
		return
	}
	if params.SampleRate < 0 || params.FrameMS < 0 || params.LevelHz < 0 || params.MaxSeconds < 0 {
		server.fail(message.ID, proto.CodeBadRequest, "Capture rates, durations and limits cannot be negative")
		return
	}
	sampleRate := params.SampleRate
	if sampleRate == 0 {
		sampleRate = 16000
	}
	frameMS := params.FrameMS
	if frameMS == 0 {
		frameMS = 100
	}
	frameSamples := sampleRate * frameMS / 1000
	if frameSamples <= 0 {
		server.fail(message.ID, proto.CodeBadRequest, "The capture frame is shorter than one sample")
		return
	}
	levelInterval := 50 * time.Millisecond
	if params.LevelHz > 0 {
		levelInterval = time.Second / time.Duration(params.LevelHz)
	}
	job, jobErr := server.registerJob(message.ID, proto.MethodCaptureStart)
	if jobErr != nil {
		server.send(proto.NewError(message.ID, jobErr))
		return
	}
	session := &captureSession{
		server:        server,
		job:           job,
		id:            message.ID,
		params:        params,
		startedAt:     time.Now(),
		rate:          sampleRate,
		frameSamples:  frameSamples,
		levelInterval: levelInterval,
		peak:          -120,
	}
	go session.run()
}

func (server *helperServer) stopCapture(message proto.Message) {
	var params proto.CaptureStopParams
	if err := message.DecodeParams(&params); err != nil || params.Job == "" {
		server.fail(message.ID, proto.CodeBadRequest, "Missing or invalid job")
		return
	}
	server.jobsMu.Lock()
	job := server.jobs[params.Job]
	server.jobsMu.Unlock()
	if job == nil || job.method != proto.MethodCaptureStart {
		server.result(message.ID, proto.CaptureStopResult{Stopped: false})
		return
	}
	job.requestStop()
	<-job.done
	server.result(message.ID, proto.CaptureStopResult{Stopped: true})
}

func (session *captureSession) run() {
	cpuStart := processCPUSeconds()
	var reason string
	var runErr *proto.Error
	if path := os.Getenv("VOXTNEW_HELPER_CAPTURE_FILE"); path != "" {
		fmt.Fprintln(os.Stderr, "capture input is a test file")
		reason, runErr = session.runFile(path)
	} else {
		runtime.LockOSThread()
		if err := windows.CoInitializeEx(0, windows.COINIT_MULTITHREADED); err != nil {
			runErr = proto.Errorf(proto.CodeInternal, "Could not initialize WASAPI: %v", err)
		} else {
			reason, runErr = session.runMicrophone()
			windows.CoUninitialize()
		}
		runtime.UnlockOSThread()
	}
	switch {
	case session.job.cancelled():
		session.server.fail(session.jobID(), proto.CodeCancelled, "The recording was cancelled")
	case runErr != nil:
		session.server.send(proto.NewError(session.jobID(), runErr))
	default:
		result := session.result(reason)
		result.CPUSeconds = math.Max(0, processCPUSeconds()-cpuStart)
		session.server.result(session.jobID(), result)
	}
	session.server.finishJob(session.jobID(), session.job)
}

func (session *captureSession) jobID() string {
	return session.id
}

func (session *captureSession) startSource(device proto.CaptureDevice) {
	session.current = device
	session.resampler.reset(device.SampleRate, session.rate)
	session.lastLevel = time.Time{}
}

func (session *captureSession) accept(samples []float64) {
	if !session.started {
		session.started = true
		session.server.notify(proto.MethodCaptureStarted, proto.CaptureStartedParams{
			Job:              session.jobID(),
			Device:           session.current.ID,
			DeviceName:       session.current.Name,
			FellBack:         session.fellBack,
			FallbackReason:   session.fallbackReason,
			DeviceSampleRate: session.current.SampleRate,
			DeviceChannels:   session.current.Channels,
			SampleRate:       session.rate,
			StartMS:          millisecondsSince(session.startedAt),
		})
	}
	output := session.resampler.add(samples)
	_, peak := audioLevels(samples)
	session.peak = math.Max(session.peak, peak)
	if session.lastLevel.IsZero() || time.Since(session.lastLevel) >= session.levelInterval {
		rms, levelPeak := audioLevels(samples)
		seconds := float64(session.totalSamples+len(session.framePending)+len(output)) / float64(session.rate)
		session.server.notify(proto.MethodCaptureLevel, proto.LevelParams{
			Job: session.jobID(), Seconds: seconds, RMS: rms, Peak: levelPeak,
		})
		session.lastLevel = time.Now()
	}
	session.framePending = append(session.framePending, output...)
	for len(session.framePending) >= session.frameSamples {
		session.emitFrame(session.framePending[:session.frameSamples])
		copy(session.framePending, session.framePending[session.frameSamples:])
		session.framePending = session.framePending[:len(session.framePending)-session.frameSamples]
	}
}

func (session *captureSession) emitFrame(samples []float64) {
	if session.sequence == 0 {
		session.firstFrameMS = millisecondsSince(session.startedAt)
	}
	data := pcm16Bytes(samples)
	session.server.notify(proto.MethodCapturePCM, proto.PCMParams{
		Job: session.jobID(), Seq: session.sequence, Offset: session.totalSamples, Data: data,
	})
	session.sequence++
	session.totalSamples += len(samples)
}

func (session *captureSession) maxSecondsReached() bool {
	return session.params.MaxSeconds > 0 && float64(session.totalSamples)/float64(session.rate) >= session.params.MaxSeconds
}

func (session *captureSession) result(reason string) proto.CaptureResult {
	if len(session.framePending) > 0 {
		session.emitFrame(session.framePending)
		session.framePending = nil
	}
	return proto.CaptureResult{
		Device:       session.current.ID,
		Reason:       reason,
		Frames:       session.sequence,
		Samples:      session.totalSamples,
		Seconds:      float64(session.totalSamples) / float64(session.rate),
		Restarts:     session.restarts,
		Peak:         session.peak,
		FirstFrameMS: session.firstFrameMS,
	}
}

func (session *captureSession) runFile(path string) (string, *proto.Error) {
	samples, format, err := readCaptureFile(path, session.rate)
	if err != nil {
		return "", proto.Errorf(proto.CodeInternal, "Could not read capture file: %v", err)
	}
	name := filepath.Base(path)
	session.startSource(proto.CaptureDevice{ID: "file:" + name, Name: name, Channels: format.channels, SampleRate: format.sampleRate})
	speed := 1.0
	if value := os.Getenv("VOXTNEW_HELPER_CAPTURE_SPEED"); value != "" {
		parsed, parseErr := strconv.ParseFloat(value, 64)
		if parseErr != nil || parsed <= 0 {
			return "", proto.Errorf(proto.CodeBadRequest, "Invalid VOXTNEW_HELPER_CAPTURE_SPEED %q", value)
		}
		speed = parsed
	}
	chunkFrames := max(1, format.sampleRate/50)
	ticker := time.NewTicker(time.Duration(float64(20*time.Millisecond) / speed))
	defer ticker.Stop()
	position := 0
	for {
		select {
		case <-session.job.stop:
			return proto.CaptureStopped, nil
		case <-session.job.cancel:
			return "", nil
		case <-ticker.C:
		}
		chunk := make([]float64, chunkFrames)
		if position < len(samples) {
			copy(chunk, samples[position:min(position+chunkFrames, len(samples))])
			position += chunkFrames
		}
		session.accept(chunk)
		if session.maxSecondsReached() {
			return proto.CaptureMaxSeconds, nil
		}
	}
}

func processCPUSeconds() float64 {
	process, _ := windows.GetCurrentProcess()
	var creation, exit, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(process, &creation, &exit, &kernel, &user); err != nil {
		return 0
	}
	return float64(kernel.Nanoseconds()+user.Nanoseconds()) / float64(time.Second)
}

func deviceUnavailable(device, message string) *proto.Error {
	data, _ := json.Marshal(map[string]string{"device": device})
	return &proto.Error{Code: proto.CodeDeviceUnavailable, Message: message, Data: data}
}
