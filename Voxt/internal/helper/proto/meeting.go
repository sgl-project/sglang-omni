package proto

// MeetingWindowParams reads a bounded Accessibility tree. The caller selects
// a process already confirmed to be a meeting, never the arbitrary front app.
type MeetingWindowParams struct {
	PID        int  `json:"pid"`
	Screenshot bool `json:"screenshot,omitempty"`
}

// MeetingWindowResult contains only Accessibility text and structure, no image.
// Empty Nodes means unavailable, denied, hidden, or outside the read budget.
type MeetingWindowResult struct {
	Nodes []MeetingNode       `json:"nodes"`
	Tiles []MeetingScreenTile `json:"tiles,omitempty"`
}

type MeetingNode struct {
	Parent      int    `json:"parent"`
	Role        string `json:"role"`
	Title       string `json:"title,omitempty"`
	Description string `json:"description,omitempty"`
	Value       string `json:"value,omitempty"`
	Identifier  string `json:"identifier,omitempty"`
	URL         string `json:"url,omitempty"`
}

// MeetingScreenTile is local OCR plus a visible speaking outline, never an image.
type MeetingScreenTile struct {
	Name     string `json:"name"`
	Speaking bool   `json:"speaking"`
}
