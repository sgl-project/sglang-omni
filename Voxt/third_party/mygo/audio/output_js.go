package audio

import (
	"errors"
	"fmt"
	"syscall/js"
	"unsafe"
)

// Browsers play through an AudioWorklet, which asks the page for a chunk
// whenever it used one up. One AudioContext serves the page for its life:
// closing the output suspends it. Pages may start audio only after the
// user interacted with them, so it resumes on the first input too.

const (
	jsFrames = 1024 // per chunk: 21 ms
	jsChunks = 3    // queued in the worklet
)

const worklet = `
class MyGoOutput extends AudioWorkletProcessor {
	constructor() {
		super();
		this.queue = [];
		this.offset = 0;
		this.port.onmessage = (e) => this.queue.push(e.data);
		for (let i = 0; i < %d; i++) this.port.postMessage(0);
	}
	process(inputs, outputs) {
		const left = outputs[0][0], right = outputs[0][1];
		let i = 0;
		while (i < left.length) {
			const chunk = this.queue[0];
			if (!chunk) {
				left.fill(0, i);
				right.fill(0, i);
				break;
			}
			const n = Math.min(left.length - i, chunk.length / 2 - this.offset);
			for (let j = 0; j < n; j++) {
				left[i + j] = chunk[2 * (this.offset + j)];
				right[i + j] = chunk[2 * (this.offset + j) + 1];
			}
			i += n;
			this.offset += n;
			if (this.offset * 2 >= chunk.length) {
				this.queue.shift();
				this.offset = 0;
				this.port.postMessage(0);
			}
		}
		return true;
	}
}
registerProcessor("mygo-output", MyGoOutput);
`

var (
	jsContext js.Value
	jsPort    js.Value
	jsCredits = make(chan struct{}, 16)
	jsErr     error
)

// jsSetup creates the page's AudioContext and its worklet, once.
func jsSetup() error {
	if !jsContext.IsUndefined() || jsErr != nil {
		return jsErr
	}
	class := js.Global().Get("AudioContext")
	if !class.Truthy() {
		jsErr = errors.New("audio: this browser has no AudioContext")
		return jsErr
	}
	ctx := class.New(map[string]any{"sampleRate": sampleRate, "latencyHint": "interactive"})
	if !ctx.Get("audioWorklet").Truthy() {
		jsErr = errors.New("audio: this browser has no AudioWorklet")
		return jsErr
	}
	blob := js.Global().Get("Blob").New([]any{fmt.Sprintf(worklet, jsChunks)}, map[string]any{"type": "text/javascript"})
	url := js.Global().Get("URL").Call("createObjectURL", blob)
	done := make(chan error, 1)
	var ok, fail js.Func
	ok = js.FuncOf(func(this js.Value, args []js.Value) any {
		done <- nil
		return nil
	})
	fail = js.FuncOf(func(this js.Value, args []js.Value) any {
		done <- fmt.Errorf("audio: loading the AudioWorklet failed: %s", js.Global().Get("String").Invoke(args[0]).String())
		return nil
	})
	ctx.Get("audioWorklet").Call("addModule", url).Call("then", ok, fail)
	err := <-done
	ok.Release()
	fail.Release()
	js.Global().Get("URL").Call("revokeObjectURL", url)
	if err != nil {
		jsErr = err
		return err
	}
	node := js.Global().Get("AudioWorkletNode").New(ctx, "mygo-output", map[string]any{
		"numberOfInputs":     0,
		"outputChannelCount": []any{channels},
	})
	node.Call("connect", ctx.Get("destination"))
	jsPort = node.Get("port")
	jsPort.Set("onmessage", js.FuncOf(func(this js.Value, args []js.Value) any {
		select {
		case jsCredits <- struct{}{}:
		default:
		}
		return nil
	}))
	// The first input lets a page start audio.
	resume := js.FuncOf(func(this js.Value, args []js.Value) any {
		if jsContext.Get("state").String() == "suspended" && jsOpen {
			jsContext.Call("resume")
		}
		return nil
	})
	for _, ev := range []string{"pointerdown", "keydown", "touchend"} {
		js.Global().Call("addEventListener", ev, resume, map[string]any{"capture": true})
	}
	jsContext = ctx
	return nil
}

// jsOpen is whether the output is open, which keeps the context running.
var jsOpen bool

type jsDevice struct{}

func openDevice() (device, int, error) {
	if err := jsSetup(); err != nil {
		return nil, 0, err
	}
	jsOpen = true
	jsContext.Call("resume")
	return jsDevice{}, jsFrames, nil
}

func (jsDevice) write(samples []float32) error {
	// A suspended context, until an input lets the page play, asks for
	// nothing: the output waits without mixing ahead.
	<-jsCredits
	a := js.Global().Get("Float32Array").New(len(samples))
	js.CopyBytesToJS(js.Global().Get("Uint8Array").New(a.Get("buffer")), unsafe.Slice((*byte)(unsafe.Pointer(&samples[0])), len(samples)*4))
	jsPort.Call("postMessage", a, []any{a.Get("buffer")})
	return nil
}

func (jsDevice) close() {
	jsOpen = false
	jsContext.Call("suspend")
}
