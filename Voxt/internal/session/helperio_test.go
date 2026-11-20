package session

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// oneCall answers one helper call with result and keeps what was asked.
type oneCall struct {
	id, method string
	params     any
	result     string
}

func (c *oneCall) Call(_ context.Context, id, method string, params any, _ func(proto.Message)) (json.RawMessage, error) {
	c.id, c.method, c.params = id, method, params
	return json.RawMessage(c.result), nil
}

// TestHelperAudioMethod: HelperAudio records the microphone with
// capture.start and the params as given, the system's sound with
// capture.system and the params it takes; both as job id.
func TestHelperAudioMethod(t *testing.T) {
	p := proto.CaptureParams{Devices: []string{"AirPodsPro:input"}, SampleRate: 16000, FrameMS: 100, LevelHz: 20, MaxSeconds: 60}
	for _, tc := range []struct {
		method, want string
		params       any
	}{
		{"", proto.MethodCaptureStart, p},
		{proto.MethodCaptureSystem, proto.MethodCaptureSystem, proto.SystemParams{SampleRate: 16000, FrameMS: 100, LevelHz: 20, MaxSeconds: 60}},
	} {
		c := &oneCall{result: `{"device":"BuiltInSpeakerDevice","reason":"stopped","frames":2,"samples":3200,"seconds":0.2,"restarts":0,"peak":-3,"cpuSeconds":0.01}`}
		res, err := HelperAudio{H: c, Method: tc.method}.Record(context.Background(), "m1/system/1", p, func(AudioEvent) {})
		if err != nil || res.Samples != 3200 || c.id != "m1/system/1" || c.method != tc.want || !reflect.DeepEqual(c.params, tc.params) {
			t.Errorf("Method %q: %+v %v; asked %s as %q with %+v", tc.method, res, err, c.method, c.id, c.params)
		}
	}
}

// TestHelperTextRead: Read asks the helper for textio.read with the
// params as given and returns its answer.
func TestHelperTextRead(t *testing.T) {
	c := &oneCall{result: `{"text":"Claude","range":{"location":3,"length":6},"length":20,"element":7}`}
	p := proto.ReadParams{PID: 812, Range: proto.TextRange{Location: 3, Length: 6}}
	got, err := HelperText{H: c}.Read(context.Background(), p)
	want := proto.ReadResult{Text: "Claude", Range: proto.TextRange{Location: 3, Length: 6}, Length: 20, Element: 7}
	if err != nil || got != want || c.method != proto.MethodTextRead || c.params != p {
		t.Errorf("Read: %+v %v; asked %s %+v", got, err, c.method, c.params)
	}
}
