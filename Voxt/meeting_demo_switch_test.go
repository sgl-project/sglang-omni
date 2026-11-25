package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// Back and Continue switch the fixture while the frame that handles the
// click is still drawing, so the content of that frame must follow the new
// fixture: between two kinds, the old kind's view read the new fixture's
// missing state and panicked (appenhance to card).
func TestMeetingDemoSwitchesAcrossFixtureKinds(t *testing.T) {
	previousLocale := i18n.Locale()
	t.Cleanup(func() { i18n.SetLocale(previousLocale) })
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	count, switched := len(host.fixtures), 0
	for index := range count {
		next := (index + 1) % count
		if host.fixtures[index].state.Kind == host.fixtures[next].state.Kind {
			continue
		}
		for _, step := range []struct {
			label    string
			from, to int
		}{{"Continue", index, next}, {"Back", next, index}} {
			host.selectMeetingFixture(step.from - host.selected)
			tester := newMeetingDemoTester(t, host)
			// A modal dialog can hide the header's button, and a fixture's own control
			// can share its label (a card's Continue), so a click may not switch at all.
			if tester.ClickControl(i18n.T(step.label)) != nil {
				continue
			}
			frames(tester)
			if host.selected == step.from {
				continue
			}
			switched++
			if host.selected != step.to {
				t.Fatalf("%s on %s selected %s, want %s", step.label, host.fixtures[step.from].name, host.current().name, host.fixtures[step.to].name)
			}
		}
	}
	if switched == 0 {
		t.Fatal("no Back or Continue switched the fixture at a boundary between fixture kinds")
	}
}
