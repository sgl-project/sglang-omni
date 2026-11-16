//go:build meetingdemo

package main

import "github.com/egoist/mygo"

func runMeetingDemo(directory string) error {
	host, err := loadMeetingDemoHost(directory)
	if err != nil {
		return err
	}
	mygo.App.WhenReady(func() {
		mygo.NewWindow(mygo.WindowOptions{
			Title: "Voxt UI Demo", Width: 1000, Height: 700,
			Content: meetingDemoContent(host),
		})
	})
	mygo.App.Run()
	return nil
}
