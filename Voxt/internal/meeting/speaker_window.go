package meeting

import (
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// SpeakerObservation contains explicit meeting UI evidence. Audio connection,
// mute state, host status and the meeting title are never speaking evidence.
type SpeakerObservation struct {
	At     *time.Duration
	Self   string
	Active []string
}

// WindowSpeakers interprets supported Accessibility layouts. Unknown layouts
// return no names; arbitrary page text and chat messages cannot name a speaker.
func WindowSpeakers(nodes []proto.MeetingNode, platform string) SpeakerObservation {
	var out SpeakerObservation
	roots := map[int]bool{}
	for i, n := range nodes {
		if platform == "zoom" && n.Identifier == "zm.meeting.window.main" {
			roots[i] = true
		}
		if platform == "meet" && n.Role == "AXWebArea" && isMeetURL(n.URL) {
			roots[i] = true
		}
	}
	if len(roots) > 1 {
		return out
	}
	inside := func(i int) bool {
		for steps := 0; i >= 0 && i < len(nodes) && steps < len(nodes); steps++ {
			if roots[i] {
				return true
			}
			next := nodes[i].Parent
			if next >= i {
				return false
			}
			i = next
		}
		return false
	}
	active := map[string]int{}
	for i, n := range nodes {
		// Zoom exposes its signed-in name on the account menu of its home window.
		if platform == "zoom" && n.Role == "AXMenuButton" && strings.HasPrefix(n.Description, "Zoom, ") && strings.HasSuffix(n.Description, " account") {
			fields := strings.Split(n.Description, ",")
			if len(fields) >= 4 {
				out.Self = cleanSpeakerName(fields[1])
			}
		}
		if !inside(i) {
			continue
		}
		if platform == "zoom" && n.Role == "AXVideoRender" {
			fields := strings.Split(n.Description, ",")
			if len(fields) < 2 {
				continue
			}
			name := cleanSpeakerName(fields[0])
			if name == "" {
				continue
			}
			if selfLabel(fields[0]) {
				out.Self = name
			}
			for _, status := range fields[1:] {
				if speakingStatus(status) {
					active[name]++
					break
				}
			}
		}
		if platform == "meet" {
			// Meet's accessible live announcement explicitly names the current speaker.
			// Restrict to descriptions, never captions/chat/static page text.
			for _, text := range []string{n.Description, n.Title} {
				if n.Role != "AXGroup" && n.Role != "AXImage" {
					continue
				}
				if name, ok := strings.CutSuffix(text, " is speaking"); ok {
					if name = cleanSpeakerName(name); name != "" {
						active[name]++
					}
				}
				if name, ok := strings.CutSuffix(text, "正在讲话"); ok {
					if name = cleanSpeakerName(name); name != "" {
						active[name]++
					}
				}
				if selfLabel(text) && !strings.ContainsAny(text, ",，\n") {
					if name := cleanSpeakerName(text); name != "" {
						out.Self = name
					}
				}
			}
		}
	}
	for name, count := range active {
		if count == 1 {
			out.Active = append(out.Active, name)
		}
	}
	return out
}

func speakingStatus(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "talking", "speaking", "is speaking", "正在讲话", "正在说话", "正在发言":
		return true
	}
	return false
}

func selfLabel(s string) bool {
	for _, suffix := range []string{" (You)", " (you)", " (Me)", "（我）", " (我)", "（您）"} {
		if strings.HasSuffix(strings.TrimSpace(s), suffix) {
			return true
		}
	}
	return false
}

func cleanSpeakerName(s string) string {
	s = strings.TrimSpace(s)
	for _, suffix := range []string{" (You)", " (you)", " (Me)", "（我）", " (我)", "（您）"} {
		s = strings.TrimSuffix(s, suffix)
	}
	s = strings.TrimSpace(s)
	if s == "" || len([]rune(s)) > 80 || strings.ContainsAny(s, "\n\r\t") {
		return ""
	}
	return s
}

func isMeetURL(s string) bool {
	// A window must identify the meeting site, not just mention it in text.
	s = strings.TrimPrefix(s, "https://")
	if !strings.HasPrefix(s, "meet.google.com/") {
		return false
	}
	path := strings.TrimPrefix(s, "meet.google.com/")
	return len(path) >= 12 && path[3] == '-' && path[8] == '-'
}

// ScreenSpeakers combines explicit Accessibility evidence with local OCR. A
// name read from a tile is only active when its speaking outline was visible.
func ScreenSpeakers(window proto.MeetingWindowResult, platform string) SpeakerObservation {
	out := WindowSpeakers(window.Nodes, platform)
	if platform != "zoom" {
		return out
	}
	var active []string
	for _, tile := range window.Tiles {
		if tile.Speaking {
			if name := cleanSpeakerName(tile.Name); name != "" {
				active = append(active, name)
			}
		}
	}
	if len(active) > 0 {
		out.Active = active
	}
	return out
}
