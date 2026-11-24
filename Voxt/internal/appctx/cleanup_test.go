package appctx

import (
	"strings"
	"testing"
)

func TestTranscribeCleanupDocument(t *testing.T) {
	for _, tc := range []struct {
		name, field string
		want        Cleanup
	}{
		{"existing group", "", CleanupDefault},
		{"on", `,"transcribeCleanup":"on"`, CleanupOn},
		{"off", `,"transcribeCleanup":"off"`, CleanupOff},
		{"invalid", `,"transcribeCleanup":"unknown"`, CleanupDefault},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, err := Parse([]byte(`{"version":1,"groups":[{"id":"g","enabled":true,"features":{"transcribe":true}` + tc.field + `}]}`))
			if err != nil || len(r.Groups) != 1 || r.Groups[0].TranscribeCleanup != tc.want {
				t.Fatalf("parsed %+v, %v", r, err)
			}
			data, err := r.Marshal()
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(data), "transcribeCleanup") != (tc.want != CleanupDefault) {
				t.Errorf("default override should be omitted: %s", data)
			}
			roundtrip, err := Parse(data)
			if err != nil || roundtrip.Groups[0].TranscribeCleanup != tc.want {
				t.Fatalf("round trip %+v, %v", roundtrip, err)
			}
		})
	}
	for _, global := range []bool{false, true} {
		if CleanupDefault.Enabled(global) != global || Cleanup("unknown").Enabled(global) != global ||
			!CleanupOn.Enabled(global) || CleanupOff.Enabled(global) {
			t.Errorf("override resolution with global %v", global)
		}
	}
	if NewGroup("g", "New").TranscribeCleanup != CleanupDefault {
		t.Error("new groups must inherit the global choice")
	}
}
