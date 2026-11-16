package main

type meetingFixtureState struct {
	Name    string                   `json:"name"`
	Kind    string                   `json:"kind"`
	Overlay *meetingOverlayViewState `json:"overlay,omitempty"`
	Page    *meetingsViewState       `json:"page,omitempty"`
	Start   *meetingStartViewState   `json:"start,omitempty"`
}
