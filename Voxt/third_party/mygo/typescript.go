//go:build !js

package mygo

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"slices"

	"github.com/egoist/mygo/internal/tsgen"
)

// GenerateTypeScript renders the typed TypeScript client for all services
// bound with Bind and events declared with NewEvent.
//
// You rarely call this directly: `mygo generate` runs your app with the
// MYGO_GENERATE environment variable set, which makes App.Run write the
// client and return before opening any window.
func GenerateTypeScript() ([]byte, error) {
	var model tsgen.Model
	ipc.RLock()
	for _, s := range ipc.services {
		if s.internal {
			continue
		}
		svc := tsgen.Service{Name: s.name, Type: s.typ}
		for _, m := range s.methods {
			meth := tsgen.Method{
				Name:     m.name,
				Params:   m.params,
				Variadic: m.variadic,
				Result:   m.result,
				HasCtx:   m.ctx,
				PC:       m.pc,
			}
			if m.streams {
				// The client declares what channels carry.
				meth.Params, meth.Channels = slices.Clone(m.params), m.chans
				for i, isChan := range m.chans {
					if isChan {
						meth.Params[i] = reflect.Zero(m.params[i]).Interface().(channelParam).valueType()
					}
				}
			}
			svc.Methods = append(svc.Methods, meth)
		}
		model.Services = append(model.Services, svc)
	}
	for _, e := range ipc.events {
		model.Events = append(model.Events, tsgen.Event{Name: e.name, Type: e.typ, PC: e.pc, File: e.file, Line: e.line})
	}
	ipc.RUnlock()
	return tsgen.Generate(model)
}

// WriteTypeScript writes the TypeScript client to path, creating parent
// directories. The file is left untouched when its content did not change,
// so frontend dev servers do not reload needlessly. An app that binds no
// services and declares no events, as one of native UI, has nothing for a
// frontend to call: it gets no client, and path is not written.
func WriteTypeScript(path string) error {
	if !bindsAnything() {
		return nil
	}
	src, err := GenerateTypeScript()
	if err != nil {
		return err
	}
	src = append(src, '\n')
	if old, err := os.ReadFile(path); err == nil && bytes.Equal(old, src) {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, src, 0o644)
}

// writeClient writes the TypeScript client to path for App.Run under
// MYGO_GENERATE, through what Bind, BindAs and NewEvent link: nothing for
// an app that binds nothing.
func writeClient(path string) error {
	ipc.RLock()
	write := ipc.writeClient
	ipc.RUnlock()
	if write == nil {
		return nil
	}
	return write(path)
}

// bindsAnything reports whether the app bound a service of its own with
// Bind or declared an event with NewEvent, for a client to call or hear.
func bindsAnything() bool {
	ipc.RLock()
	defer ipc.RUnlock()
	if len(ipc.events) > 0 {
		return true
	}
	for _, s := range ipc.services {
		if !s.internal {
			return true
		}
	}
	return false
}
