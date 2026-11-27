//go:build !meetingdemo

package main

import (
	"strings"
	"testing"
)

func TestMeetingDemoRequiresBuildTag(t *testing.T) {
	if err := runMeetingDemo("testdata/meeting"); err == nil || !strings.Contains(err.Error(), "meetingdemo") {
		t.Fatalf("demo without tag: %v", err)
	}
}
