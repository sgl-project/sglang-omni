//go:build windows

package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

func TestSelectionDecision(t *testing.T) {
	tests := []struct {
		name      string
		probe     selectionProbe
		allowCopy bool
		want      selectionDecision
	}{
		{
			name:  "UIA range wins over copy",
			probe: selectionProbe{available: true, hasSelection: true, text: "selected"},
			want:  selectionDecision{text: "selected", via: proto.ViaAXRange},
		},
		{
			name:      "caret never copies",
			probe:     selectionProbe{available: true},
			allowCopy: true,
			want:      selectionDecision{via: proto.ViaNone},
		},
		{
			name:      "unknown selection never copies",
			allowCopy: true,
			want:      selectionDecision{via: proto.ViaNone},
		},
		{
			name:      "unreadable nonempty selection can copy",
			probe:     selectionProbe{available: true, hasSelection: true},
			allowCopy: true,
			want:      selectionDecision{via: proto.ViaNone, copyNext: true},
		},
		{
			name:  "copy policy belongs to caller",
			probe: selectionProbe{available: true, hasSelection: true},
			want:  selectionDecision{via: proto.ViaNone},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := decideSelection(test.probe, test.allowCopy); got != test.want {
				t.Fatalf("decideSelection() = %+v, want %+v", got, test.want)
			}
		})
	}
}
