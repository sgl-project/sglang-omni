//go:build meetingdemo

package main

import (
	"os"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func runMeetingDemo(directory string) error {
	host, err := loadMeetingDemoHost(directory)
	if err != nil {
		return err
	}
	if os.Getenv("MYGO_LOCALE") == "" {
		if err := os.Setenv("MYGO_LOCALE", host.current().locale); err != nil {
			return err
		}
	}
	mygo.App.SetLanguages(i18n.Supported()...)
	mygo.App.WhenReady(func() {
		mygo.NewWindow(mygo.WindowOptions{
			Title: "Voxt UI Demo", Width: 1000, Height: 700,
			Content: meetingDemoContent(host, func() { showLanguage(host.current().locale) }),
		})
	})
	return mygo.App.Run()
}
