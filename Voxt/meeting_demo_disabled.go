//go:build !meetingdemo

package main

import "fmt"

func runMeetingDemo(directory string) error {
	return fmt.Errorf("VOXTNEW_DEMO_DIR=%s requires -tags=meetingdemo", directory)
}
