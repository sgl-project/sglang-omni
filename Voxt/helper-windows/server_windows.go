//go:build windows

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

type helperJob struct {
	method     string
	cancel     chan struct{}
	stop       chan struct{}
	done       chan struct{}
	cancelOnce sync.Once
	stopOnce   sync.Once
}

func (job *helperJob) requestCancel() {
	job.cancelOnce.Do(func() { close(job.cancel) })
}

func (job *helperJob) requestStop() {
	job.stopOnce.Do(func() { close(job.stop) })
}

func (job *helperJob) stopped() bool {
	select {
	case <-job.stop:
		return true
	default:
		return false
	}
}

func (job *helperJob) cancelled() bool {
	select {
	case <-job.cancel:
		return true
	default:
		return false
	}
}

type helperServer struct {
	output      *bufio.Writer
	writeMu     sync.Mutex
	initialized atomic.Bool
	started     time.Time
	shutdown    chan struct{}
	stopOnce    sync.Once

	jobsMu sync.Mutex
	jobs   map[string]*helperJob

	events     *windowsEventLoop
	automation *automationThread
	hotkeys    *windowsHotkeys

	frontMu    sync.RWMutex
	frontCache proto.App
}

func implementedMethods() []string {
	return []string{
		proto.MethodHello,
		proto.MethodShutdown,
		proto.MethodJobCancel,
		proto.MethodPermissionsStatus,
		proto.MethodPermissionsRequest,
		proto.MethodAppFrontmost,
		proto.MethodCaptureDevices,
		proto.MethodCaptureStart,
		proto.MethodCaptureStop,
		proto.MethodHotkeyConfigure,
		proto.MethodHotkeyStatus,
		proto.MethodTextFocus,
		proto.MethodTextPaste,
	}
}

func runHelper() error {
	server := &helperServer{
		output:   bufio.NewWriter(os.Stdout),
		started:  time.Now(),
		shutdown: make(chan struct{}),
		jobs:     make(map[string]*helperJob),
	}
	events, err := startWindowsEventLoop(server.foregroundChanged)
	if err != nil {
		return err
	}
	server.events = events
	defer events.close()
	automation, err := startAutomationThread()
	if err != nil {
		return err
	}
	server.automation = automation
	defer automation.close()
	server.hotkeys = newWindowsHotkeys(server)

	messages := make(chan proto.Message)
	readErrors := make(chan error, 1)
	go scanRequests(messages, readErrors)
	interrupts := make(chan os.Signal, 1)
	signal.Notify(interrupts, os.Interrupt)
	defer signal.Stop(interrupts)

	for {
		select {
		case message := <-messages:
			if message.Method == proto.MethodHello {
				server.serve(message)
			} else {
				go server.serve(message)
			}
		case err := <-readErrors:
			server.stopOnce.Do(func() { close(server.shutdown) })
			server.cancelAllJobs()
			return err
		case <-interrupts:
			server.stopOnce.Do(func() { close(server.shutdown) })
			server.cancelAllJobs()
			return nil
		case <-server.shutdown:
			return nil
		}
	}
}

func scanRequests(messages chan<- proto.Message, readErrors chan<- error) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 64*1024), proto.MaxLine)
	for scanner.Scan() {
		message, err := proto.Decode(scanner.Bytes())
		if err != nil {
			fmt.Fprintln(os.Stderr, "bad request:", err)
			continue
		}
		if message.Kind() != proto.KindRequest {
			fmt.Fprintln(os.Stderr, "ignored non-request wire message")
			continue
		}
		messages <- message
	}
	readErrors <- scanner.Err()
}

func (server *helperServer) send(message proto.Message) {
	line, err := proto.Encode(message)
	if err != nil {
		fmt.Fprintln(os.Stderr, "encode response:", err)
		return
	}
	server.writeMu.Lock()
	defer server.writeMu.Unlock()
	if _, err := server.output.Write(line); err != nil {
		fmt.Fprintln(os.Stderr, "write response:", err)
		return
	}
	if err := server.output.Flush(); err != nil {
		fmt.Fprintln(os.Stderr, "flush response:", err)
	}
}

func (server *helperServer) result(id string, result any) {
	message, err := proto.NewResult(id, result)
	if err != nil {
		server.fail(id, proto.CodeInternal, "Could not encode result: %v", err)
		return
	}
	server.send(message)
}

func (server *helperServer) fail(id, code, format string, arguments ...any) {
	server.send(proto.NewError(id, proto.Errorf(code, format, arguments...)))
}

func (server *helperServer) notify(method string, params any) {
	message, err := proto.NewNotification(method, params)
	if err != nil {
		fmt.Fprintln(os.Stderr, "encode notification:", err)
		return
	}
	server.send(message)
}

func (server *helperServer) serve(message proto.Message) {
	if message.Method == proto.MethodHello {
		server.hello(message)
		return
	}
	if !server.initialized.Load() {
		server.fail(message.ID, proto.CodeNotInitialized, "helper.hello must come first")
		return
	}
	switch message.Method {
	case proto.MethodShutdown:
		server.cancelAllJobs()
		server.result(message.ID, proto.Empty{})
		server.stopOnce.Do(func() { close(server.shutdown) })
	case proto.MethodJobCancel:
		server.cancelJob(message)
	case proto.MethodPermissionsStatus:
		server.result(message.ID, windowsPermissions())
	case proto.MethodPermissionsRequest:
		var params proto.PermissionsRequestParams
		if err := message.DecodeParams(&params); err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
			return
		}
		switch params.Permission {
		case proto.PermissionMicrophone, proto.PermissionAccessibility, proto.PermissionSpeechRecognition:
			server.result(message.ID, windowsPermissions())
		default:
			server.fail(message.ID, proto.CodeBadRequest, "Unknown permission %s", params.Permission)
		}
	case proto.MethodAppFrontmost:
		app, err := frontmostApp()
		if err != nil {
			server.fail(message.ID, proto.CodeInternal, "Could not read the foreground app: %v", err)
			return
		}
		server.result(message.ID, app)
	case proto.MethodCaptureDevices:
		server.serveCaptureDevices(message)
	case proto.MethodCaptureStart:
		server.startCapture(message)
	case proto.MethodCaptureStop:
		server.stopCapture(message)
	case proto.MethodHotkeyConfigure:
		var params proto.HotkeyConfigureParams
		if err := message.DecodeParams(&params); err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
			return
		}
		status, err := server.hotkeys.configure(params)
		if err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad hotkey configuration: %v", err)
			return
		}
		server.result(message.ID, status)
	case proto.MethodHotkeyStatus:
		server.result(message.ID, server.hotkeys.status())
	case proto.MethodTextFocus, proto.MethodTextPaste:
		server.serveTextIO(message)
	default:
		server.fail(message.ID, proto.CodeMethodNotFound, "Unknown method %s", message.Method)
	}
}

func (server *helperServer) hello(message proto.Message) {
	var params proto.HelloParams
	if err := message.DecodeParams(&params); err != nil {
		server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
		return
	}
	if params.Protocol != proto.Version {
		server.send(proto.NewError(message.ID, &proto.Error{
			Code:    proto.CodeProtocolMismatch,
			Message: "This helper speaks protocol 1",
			Data:    json.RawMessage(`{"supported":[1]}`),
		}))
		return
	}
	version := "dev"
	if build, ok := debug.ReadBuildInfo(); ok && build.Main.Version != "" && build.Main.Version != "(devel)" {
		version = build.Main.Version
	}
	hostName, _ := os.Hostname()
	osVersion := windowsVersion()
	server.initialized.Store(true)
	server.result(message.ID, proto.HelloResult{
		Protocol: proto.Version,
		Version:  version,
		PID:      os.Getpid(),
		Methods:  implementedMethods(),
		Device:   proto.Device{Name: hostName, OS: osVersion},
	})
}

func windowsPermissions() proto.PermissionsResult {
	return proto.PermissionsResult{
		Microphone:        proto.PermGranted,
		Accessibility:     true,
		SpeechRecognition: proto.PermGranted,
	}
}

func (server *helperServer) foregroundChanged(windowHandle uintptr) {
	if !server.initialized.Load() {
		return
	}
	app, err := appForWindow(windowHandle)
	if err != nil || app.PID == os.Getpid() {
		return
	}
	server.frontMu.Lock()
	server.frontCache = app
	server.frontMu.Unlock()
	server.notify(proto.MethodAppActivated, app)
}

func (server *helperServer) cachedFront() proto.App {
	server.frontMu.RLock()
	front := server.frontCache
	server.frontMu.RUnlock()
	if front.PID != 0 {
		return front
	}
	front, err := frontmostApp()
	if err == nil {
		return front
	}
	return proto.App{}
}

func (server *helperServer) registerJob(id, method string) (*helperJob, *proto.Error) {
	server.jobsMu.Lock()
	defer server.jobsMu.Unlock()
	if _, running := server.jobs[id]; running {
		return nil, proto.Errorf(proto.CodeBusy, "Job %s is running already", id)
	}
	if method == proto.MethodCaptureStart {
		for _, running := range server.jobs {
			if running.method == method {
				return nil, proto.Errorf(proto.CodeBusy, "A recording is running already")
			}
		}
	}
	job := &helperJob{method: method, cancel: make(chan struct{}), stop: make(chan struct{}), done: make(chan struct{})}
	server.jobs[id] = job
	return job, nil
}

func (server *helperServer) finishJob(id string, job *helperJob) {
	server.jobsMu.Lock()
	if server.jobs[id] == job {
		delete(server.jobs, id)
	}
	server.jobsMu.Unlock()
	close(job.done)
}

func (server *helperServer) cancelJob(message proto.Message) {
	var params proto.CancelParams
	if err := message.DecodeParams(&params); err != nil || params.Job == "" {
		server.fail(message.ID, proto.CodeBadRequest, "Missing or invalid job")
		return
	}
	server.jobsMu.Lock()
	job := server.jobs[params.Job]
	server.jobsMu.Unlock()
	if job == nil {
		server.result(message.ID, proto.CancelResult{Cancelled: false})
		return
	}
	job.requestCancel()
	<-job.done
	server.result(message.ID, proto.CancelResult{Cancelled: true})
}

func (server *helperServer) cancelAllJobs() {
	server.jobsMu.Lock()
	jobs := make([]*helperJob, 0, len(server.jobs))
	for _, job := range server.jobs {
		jobs = append(jobs, job)
	}
	server.jobsMu.Unlock()
	for _, job := range jobs {
		job.requestCancel()
	}
}
