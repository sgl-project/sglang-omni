package main

import "testing"

// The answer card's formula preview follows the card's lifecycle: a
// finished stream measures its final height, a hidden or closed card
// closes its previews, and a turn that becomes the user's drops its slot.

func TestPreviewHeightResetsWhenStreamingEnds(t *testing.T) {
	preview := &transcriptPreview{
		input:  transcriptPreviewInput{key: "card", source: "so $$x^2$$", streaming: true},
		height: 132, ready: true, loaded: true,
	}
	final := preview.input
	final.streaming = false
	if !preview.adopt(final) {
		t.Fatal("a stream that ends renders once more")
	}
	if preview.height != 0 {
		t.Fatalf("height = %d, want 0: the partial render's height must not survive the final one", preview.height)
	}
	if !preview.ready || !preview.loaded {
		t.Fatal("the shown page stays until the final render replaces it")
	}
}

func TestPreviewReopensWhenAskedAgain(t *testing.T) {
	preview := &transcriptPreview{closed: true, input: transcriptPreviewInput{key: "card", source: "one"}}
	if !preview.adopt(transcriptPreviewInput{key: "card", source: "two"}) {
		t.Fatal("a new input renders again")
	}
	if preview.closed {
		t.Fatal("a frame that asks for the preview reopens it")
	}
}

func TestBubblePreviewDropsWhenTheTurnIsTheUsers(t *testing.T) {
	a := &app{}
	slot := &transcriptPreview{
		input:  transcriptPreviewInput{key: "s/0", source: "so $$x^2$$"},
		ready:  true,
		height: 96,
	}
	a.bubbleTranscripts = []*transcriptPreview{slot}
	state := &sessionOverlayViewState{
		SessionID: "s",
		Card: &sessionCardViewState{
			Conversation: true,
			Blocks: []sessionCardBlock{
				{Kind: sessionCardBubble, User: true, Text: "again?"},
			},
		},
	}
	if heights := a.sessionOverlayPreviewHeights(state); len(heights) != 0 {
		t.Fatalf("a user bubble reports no preview height, got %v", heights)
	}
	if slot.ready || slot.height != 0 || !slot.closed {
		t.Fatalf("the slot kept its preview next to the user's bubble: ready=%v height=%d closed=%v", slot.ready, slot.height, slot.closed)
	}
	// A later size pass leaves the dropped slot alone.
	a.sessionOverlayPreviewHeights(state)
	if slot.ready || slot.height != 0 || !slot.closed {
		t.Fatalf("the dropped slot came back on its own: ready=%v height=%d closed=%v", slot.ready, slot.height, slot.closed)
	}
}

func TestHideClosesTheOverlaysPreviews(t *testing.T) {
	a := &app{}
	a.overlayTranscript = transcriptPreview{ready: true, height: 40}
	a.bubbleTranscripts = []*transcriptPreview{nil, {ready: true, height: 64}}
	a.closeOverlayPreviews()
	if !a.overlayTranscript.closed {
		t.Fatal("the answer preview stayed open when the card hid")
	}
	if !a.bubbleTranscripts[1].closed {
		t.Fatal("a conversation preview stayed open when the card hid")
	}
}

func TestCloseOverlayResetsPreviewState(t *testing.T) {
	a := &app{}
	a.overlayTranscript = transcriptPreview{ready: true, height: 40}
	a.bubbleTranscripts = []*transcriptPreview{{ready: true}}
	a.closeOverlay()
	if a.overlayTranscript.ready || a.overlayTranscript.height != 0 {
		t.Fatalf("the answer preview survived closeOverlay: %+v", a.overlayTranscript)
	}
	if a.bubbleTranscripts != nil {
		t.Fatalf("the conversation previews survived closeOverlay: %v", a.bubbleTranscripts)
	}
}
