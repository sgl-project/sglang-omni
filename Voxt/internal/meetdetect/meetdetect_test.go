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

// held is what Update says about users once they have recorded for Hold.
func held(d *Detector, users []User, browsers []string, start time.Time) []Candidate {
	d.Update(users, browsers, start)
	found, _ := d.Update(users, browsers, start.Add(Hold))
	return found
}

func TestUpdatePicksMeetingApps(t *testing.T) {
	var d Detector
	got := held(&d, []User{
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
		got := held(&d, []User{{PID: 1, BundleID: id}}, testBrowsers, time.Unix(0, 0))
		if len(got) != 1 || got[0].Browser != want || got[0].Key != want || got[0].App != "" {
			t.Errorf("%s: candidates = %+v, want the browser %s", id, got, want)
		}
	}
	var d Detector
	if got := held(&d, []User{{PID: 1, BundleID: "com.google.ChromeX"}}, testBrowsers, time.Unix(0, 0)); len(got) != 0 {
		t.Errorf("another app's id that starts like a browser's: %+v", got)
	}
}

// An app is a candidate only once it has recorded for Hold: a microphone
// test of a moment is none, and Update says when to look again.
func TestHold(t *testing.T) {
	var d Detector
	zoom := []User{{PID: 704, BundleID: "us.zoom.xos"}}
	t0 := time.Unix(1000, 0)
	if got, wait := d.Update(zoom, nil, t0); len(got) != 0 || wait != Hold {
		t.Fatalf("just started: %v, look again in %v", keys(got), wait)
	}
	if got, wait := d.Update(zoom, nil, t0.Add(Hold-time.Second)); len(got) != 0 || wait != time.Second {
		t.Fatalf("a second short of Hold: %v, look again in %v", keys(got), wait)
	}
	if got, wait := d.Update(nil, nil, t0.Add(Hold-time.Millisecond)); len(got) != 0 || wait != 0 {
		t.Fatalf("stopped before Hold: %v, look again in %v", keys(got), wait)
	}
	// The next use starts its own Hold.
	if got, wait := d.Update(zoom, nil, t0.Add(Hold)); len(got) != 0 || wait != Hold {
		t.Fatalf("started again: %v, look again in %v", keys(got), wait)
	}
	if got, wait := d.Update(zoom, nil, t0.Add(2*Hold)); len(got) != 1 || wait != 0 {
		t.Fatalf("held: %v, look again in %v", keys(got), wait)
	}
	short := Detector{Hold: time.Millisecond}
	short.Update(zoom, nil, t0)
	if got, _ := short.Update(zoom, nil, t0.Add(time.Millisecond)); len(got) != 1 {
		t.Fatalf("a Detector's own Hold: %v", keys(got))
	}
}

// An app the user was asked about is not asked about again while it
// records, nor at an unmute soon after; a later meeting asks again, also
// when nothing else changed in between.
func TestSettledUntilQuiet(t *testing.T) {
	var d Detector
	zoom := []User{{PID: 704, BundleID: "us.zoom.xos"}}
	t0 := time.Unix(1000, 0)
	if got := keys(held(&d, zoom, nil, t0)); len(got) != 1 {
		t.Fatalf("first use: %v", got)
	}
	d.Settle("us.zoom.xos")
	if !d.Recording("us.zoom.xos") {
		t.Fatal("Recording is false while Zoom records")
	}
	if got, _ := d.Update(zoom, nil, t0.Add(time.Minute)); len(got) != 0 {
		t.Fatalf("asked again while recording: %v", keys(got))
	}
	if got, _ := d.Update(nil, nil, t0.Add(30*time.Minute)); len(got) != 0 || d.Recording("us.zoom.xos") {
		t.Fatalf("after it stopped: %v, recording %v", keys(got), d.Recording("us.zoom.xos"))
	}
	if got := keys(held(&d, zoom, nil, t0.Add(30*time.Minute+30*time.Second))); len(got) != 0 {
		t.Fatalf("asked again at an unmute: %v", got)
	}
	d.Update(nil, nil, t0.Add(40*time.Minute))
	// Three hours later, with no Update in between: the next meeting.
	if got := keys(held(&d, zoom, nil, t0.Add(3*time.Hour))); len(got) != 1 {
		t.Fatalf("the next meeting did not ask: %v", got)
	}

	// The same for a browser settled as no meeting: an hour later its tab
	// is read again.
	var b Detector
	chrome := []User{{PID: 4764, BundleID: "com.google.Chrome.helper"}}
	if got := held(&b, chrome, testBrowsers, t0); len(got) != 1 {
		t.Fatalf("Chrome records: %+v", got)
	}
	b.Settle("com.google.Chrome")
	b.Update(nil, testBrowsers, t0.Add(time.Minute))
	if got := held(&b, chrome, testBrowsers, t0.Add(time.Hour)); len(got) != 1 {
		t.Fatalf("Chrome records an hour later: %+v", got)
	}
}

func TestForget(t *testing.T) {
	var d Detector
	zoom := []User{{PID: 704, BundleID: "us.zoom.xos"}}
	t0 := time.Unix(1000, 0)
	held(&d, zoom, nil, t0)
	d.Settle("us.zoom.xos")
	d.Forget("us.zoom.xos")
	if got, _ := d.Update(zoom, nil, t0.Add(2*time.Minute)); len(got) != 1 {
		t.Fatalf("forgotten while it records: %v", keys(got))
	}
}

func TestWeb(t *testing.T) {
	for address, want := range map[string]string{
		"https://meet.google.com/abc-defg-hij":        Meet,
		"https://teams.microsoft.com/v2/":             Teams,
		"https://app.zoom.us/wc/123/join":             Zoom,
		"https://us02web.zoom.us/j/123":               Zoom,
		"meeting.tencent.com/dm/abc":                  Tencent,
		"https://acme.webex.com/meet/ann":             Webex,
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

// Zoom records for a meeting only while one of its meeting processes runs
// for it: its settings' microphone test brings no card (11-25), and the
// meeting is ready as soon as the meeting process came. A helper that
// does not say (Hosts nil) leaves Zoom a meeting.
func TestZoomNeedsItsMeetingProcess(t *testing.T) {
	var d Detector
	start := time.Unix(0, 0)
	test := []User{{PID: 758, BundleID: "us.zoom.xos", Hosts: []string{}}}
	if found := held(&d, test, nil, start); len(found) != 0 || d.Recording("us.zoom.xos") {
		t.Fatalf("Zoom's microphone test: %+v", found)
	}
	meeting := []User{{PID: 758, BundleID: "us.zoom.xos", Hosts: []string{"aomhost"}}}
	later := start.Add(time.Minute)
	if found, wait := d.Update(meeting, nil, later); len(found) != 1 || found[0].App != Zoom || wait != 0 {
		t.Fatalf("its confirmed meeting should ask at once: %+v, wait %v", found, wait)
	}
	if found, _ := d.Update(meeting, nil, later.Add(Hold)); len(found) != 1 || found[0].App != Zoom {
		t.Fatalf("its meeting after Hold: %+v", found)
	}
	var old Detector
	if found := held(&old, []User{{PID: 758, BundleID: "us.zoom.xos"}}, nil, start); len(found) != 1 {
		t.Fatalf("a helper that does not say: %+v", found)
	}
	if got := HostNames(); len(got) != 2 || got[0] != "CptHost" || got[1] != "aomhost" {
		t.Errorf("HostNames = %q", got)
	}
}
