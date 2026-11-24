package meetdetect

import (
	"testing"
	"time"
)

var testBrowsers = []string{"com.google.Chrome", "com.google.Chrome.beta", "com.apple.Safari"}

func keys(cs []Candidate) []string {
	out := []string{}
	for _, c := range cs {
		out = append(out, c.Key)
	}
	return out
}

func TestUpdatePicksMeetingApps(t *testing.T) {
	var d Detector
	got := d.Update([]User{
		{PID: 1206, BundleID: "com.apple.CoreSpeech"}, // Siri listens all day
		{PID: 704, BundleID: "us.zoom.xos"},
		{PID: 12, BundleID: "com.tencent.xinWeChat"}, // a voice message
		{PID: 30, BundleID: "com.microsoft.teams2.helper"},
		{PID: 31, BundleID: "com.microsoft.teams2.helper"},
		{PID: 40},
	}, testBrowsers, time.Unix(0, 0))
	want := []Candidate{{Key: "com.microsoft.teams2", App: Teams}, {Key: "us.zoom.xos", App: Zoom}}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("candidates = %+v, want %+v", got, want)
	}
}

func TestUpdateNamesTheBrowser(t *testing.T) {
	for id, want := range map[string]string{
		"com.google.Chrome.helper":      "com.google.Chrome",
		"com.google.Chrome":             "com.google.Chrome",
		"com.google.Chrome.beta.helper": "com.google.Chrome.beta",
		"com.apple.WebKit.GPU":          "com.apple.Safari",
	} {
		var d Detector
		got := d.Update([]User{{PID: 1, BundleID: id}}, testBrowsers, time.Unix(0, 0))
		if len(got) != 1 || got[0].Browser != want || got[0].Key != want || got[0].App != "" {
			t.Errorf("%s: candidates = %+v, want the browser %s", id, got, want)
		}
	}
	var d Detector
	if got := d.Update([]User{{PID: 1, BundleID: "com.google.ChromeX"}}, testBrowsers, time.Unix(0, 0)); len(got) != 0 {
		t.Errorf("another app's id that starts like a browser's: %+v", got)
	}
}

// An app the user was asked about is not asked about again while it
// records, nor at an unmute soon after; a later meeting asks again.
func TestSettledUntilQuiet(t *testing.T) {
	var d Detector
	zoom := []User{{PID: 704, BundleID: "us.zoom.xos"}}
	t0 := time.Unix(1000, 0)
	if got := keys(d.Update(zoom, nil, t0)); len(got) != 1 {
		t.Fatalf("first use: %v", got)
	}
	d.Settle("us.zoom.xos")
	if !d.Recording("us.zoom.xos") {
		t.Fatal("Recording is false while Zoom records")
	}
	if got := keys(d.Update(zoom, nil, t0.Add(time.Second))); len(got) != 0 {
		t.Fatalf("asked again while recording: %v", got)
	}
	if got := keys(d.Update(nil, nil, t0.Add(10*time.Second))); len(got) != 0 || d.Recording("us.zoom.xos") {
		t.Fatalf("after it stopped: %v, recording %v", got, d.Recording("us.zoom.xos"))
	}
	if got := keys(d.Update(zoom, nil, t0.Add(40*time.Second))); len(got) != 0 {
		t.Fatalf("asked again at an unmute: %v", got)
	}
	d.Update(nil, nil, t0.Add(50*time.Second))
	d.Update(nil, nil, t0.Add(50*time.Second+Quiet+time.Second))
	if got := keys(d.Update(zoom, nil, t0.Add(time.Hour))); len(got) != 1 {
		t.Fatalf("the next meeting did not ask: %v", got)
	}
}

func TestWeb(t *testing.T) {
	for address, want := range map[string]string{
		"https://meet.google.com/abc-defg-hij":        Meet,
		"https://teams.microsoft.com/v2/":             Teams,
		"https://app.zoom.us/wc/123/join":             Zoom,
		"https://us02web.zoom.us/j/123":               Zoom,
		"meeting.tencent.com/dm/abc":                  Tencent,
		"https://docs.google.com/document/d/1":        "",
		"https://notzoom.us/":                         "",
		"https://chatgpt.com/":                        "",
		"":                                            "",
		"https://example.com/?next=meet.google.com/x": "",
	} {
		got, ok := Web(address)
		if got != want || ok != (want != "") {
			t.Errorf("Web(%q) = %q, %v; want %q", address, got, ok, want)
		}
	}
}
