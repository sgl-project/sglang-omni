//go:build windows

package main

import (
	"errors"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"golang.org/x/sys/windows"
)

const outputPollInterval = 100 * time.Millisecond

type windowsOutput struct {
	server *helperServer
	claims *outputClaims

	watchMutex sync.Mutex
	watching   bool
	lastState  proto.OutputState
	started    bool
	startOnce  sync.Once
	stop       chan struct{}
	done       chan struct{}
	stopOnce   sync.Once
}

func newWindowsOutput(server *helperServer) *windowsOutput {
	return &windowsOutput{
		server: server,
		claims: newOutputClaims(&wasapiOutputHardware{}, os.Getpid(), windowsProcessRuns),
		stop:   make(chan struct{}),
		done:   make(chan struct{}),
	}
}

func (output *windowsOutput) watch(params proto.OutputWatchParams) proto.OutputState {
	state := output.claims.state()
	output.watchMutex.Lock()
	output.watching = params.On
	output.lastState = state
	if params.On {
		output.startOnce.Do(func() {
			output.started = true
			go output.watchLoop()
		})
	}
	output.watchMutex.Unlock()
	return state
}

func (output *windowsOutput) watchLoop() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(output.done)
	if err := windows.CoInitializeEx(0, windows.COINIT_MULTITHREADED); err != nil {
		fmt.Fprintln(os.Stderr, "output watch COM initialization:", err)
		return
	}
	defer windows.CoUninitialize()
	ticker := time.NewTicker(outputPollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-output.stop:
			return
		case <-ticker.C:
			output.watchMutex.Lock()
			watching := output.watching
			output.watchMutex.Unlock()
			if !watching {
				continue
			}
			state := output.claims.state()
			output.watchMutex.Lock()
			changed := output.watching && !reflect.DeepEqual(state, output.lastState)
			if changed {
				output.lastState = state
			}
			watching = output.watching
			output.watchMutex.Unlock()
			if watching && changed {
				output.server.notify(proto.MethodOutputChanged, state)
			}
		}
	}
}

func (output *windowsOutput) close() {
	output.stopOnce.Do(func() { close(output.stop) })
	output.watchMutex.Lock()
	started := output.started
	output.watchMutex.Unlock()
	if started {
		<-output.done
	}
	if err := runOutputCOM(func() {
		released := output.claims.releaseAll()
		if len(released) > 0 {
			fmt.Fprintln(os.Stderr, "output: unmuted", len(released), "device(s) on exit")
		}
	}); err != nil {
		fmt.Fprintln(os.Stderr, "output cleanup:", err)
	}
}

func (server *helperServer) serveOutput(message proto.Message) {
	switch message.Method {
	case proto.MethodOutputWatch:
		var params proto.OutputWatchParams
		if err := message.DecodeParams(&params); err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
			return
		}
		server.serveOutputCall(message.ID, func() (any, *proto.Error) {
			return server.outputAudio.watch(params), nil
		})
	case proto.MethodOutputMute:
		var params proto.OutputMuteParams
		if err := message.DecodeParams(&params); err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
			return
		}
		server.serveOutputCall(message.ID, func() (any, *proto.Error) {
			result, outputErr := server.outputAudio.claims.mute(params)
			return result, outputErr
		})
	case proto.MethodOutputRelease:
		var params proto.OutputReleaseParams
		if err := message.DecodeParams(&params); err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
			return
		}
		server.serveOutputCall(message.ID, func() (any, *proto.Error) {
			return server.outputAudio.claims.release(params), nil
		})
	case proto.MethodOutputRecover:
		var params proto.OutputRecoverParams
		if err := message.DecodeParams(&params); err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
			return
		}
		server.serveOutputCall(message.ID, func() (any, *proto.Error) {
			return server.outputAudio.claims.recover(params), nil
		})
	}
}

func (server *helperServer) serveOutputCall(id string, action func() (any, *proto.Error)) {
	var result any
	var outputErr *proto.Error
	if err := runOutputCOM(func() { result, outputErr = action() }); err != nil {
		server.fail(id, proto.CodeInternal, "Could not initialize output audio: %v", err)
		return
	}
	if outputErr != nil {
		server.send(proto.NewError(id, outputErr))
		return
	}
	server.result(id, result)
}

func runOutputCOM(action func()) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := windows.CoInitializeEx(0, windows.COINIT_MULTITHREADED); err != nil {
		return err
	}
	defer windows.CoUninitialize()
	action()
	return nil
}

func windowsProcessRuns(pid int) bool {
	process, err := windows.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil {
		return errors.Is(err, windows.ERROR_ACCESS_DENIED)
	}
	defer windows.CloseHandle(process)
	var exitCode uint32
	if err := windows.GetExitCodeProcess(process, &exitCode); err != nil {
		return true
	}
	return exitCode == 259
}
