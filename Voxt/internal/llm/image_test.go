package llm

import (
	"context"
	"encoding/json"
	"testing"
)

// TestImageGoesWithLastUserMessage: Request.Image becomes the last user
// message's image part, in the chat completions' and the Responses API's
// format; the other messages stay text.
func TestImageGoesWithLastUserMessage(t *testing.T) {
	msgs := []Message{{Role: System, Content: "sys"}, {Role: User, Content: "one"}, {Role: Assistant, Content: "ok"}, {Role: User, Content: "two"}}
	jpeg := []byte{0xff, 0xd8, 0xff, 0xd9}
	url := "data:image/jpeg;base64,/9j/2Q=="

	f := newFakeServer(t, jsonReply(200, oaText("x")))
	if _, err := NewOpenAI(presetAt(t, "openai", f.srv.URL), "sk", f.srv.Client()).Chat(context.Background(), Request{Messages: msgs, Image: jpeg}); err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(f.requests()[0].Body["messages"])
	want := `[{"content":"sys","role":"system"},{"content":"one","role":"user"},{"content":"ok","role":"assistant"},` +
		`{"content":[{"text":"two","type":"text"},{"image_url":{"url":"` + url + `"},"type":"image_url"}],"role":"user"}]`
	if string(got) != want {
		t.Errorf("chat messages\n got %s\nwant %s", got, want)
	}

	f = newFakeServer(t, jsonReply(200, rsObject("completed", []any{rsMessage(rsTextPart("ok"))}, rsUsed, nil, nil)))
	if _, err := NewOpenAI(rsPreset(t, "openai", f.srv.URL), "sk", f.srv.Client()).Chat(context.Background(), Request{Messages: msgs, Image: jpeg}); err != nil {
		t.Fatal(err)
	}
	b := f.requests()[0].Body
	got, _ = json.Marshal(b["input"])
	want = `[{"content":"one","role":"user"},{"content":"ok","role":"assistant"},` +
		`{"content":[{"text":"two","type":"input_text"},{"image_url":"` + url + `","type":"input_image"}],"role":"user"}]`
	if string(got) != want || b["instructions"] != "sys" {
		t.Errorf("responses input\n got %s\nwant %s (instructions %v)", got, want, b["instructions"])
	}
}

// TestImageRefusedSendsText: a model that takes no images refuses the
// request; it is sent again, once, with the text alone, and answers.
func TestImageRefusedSendsText(t *testing.T) {
	f := newFakeServer(t,
		jsonReply(400, map[string]any{"error": map[string]any{"message": "Invalid content type. image_url is only supported by certain models."}}),
		jsonReply(200, oaText("fine")),
	)
	c := &Client{Backend: NewOpenAI(presetAt(t, "openai", f.srv.URL), "sk", f.srv.Client())}
	resp, err := c.Chat(context.Background(), Request{Messages: msgs("hi"), Image: []byte{0xff, 0xd8}})
	if err != nil || resp.Text != "fine" {
		t.Fatalf("%v %+v", err, resp)
	}
	rs := f.requests()
	if len(rs) != 2 {
		t.Fatalf("%d requests", len(rs))
	}
	if _, parts := rs[0].Body["messages"].([]any)[0].(map[string]any)["content"].([]any); !parts {
		t.Errorf("first request without the image: %v", rs[0].Body["messages"])
	}
	if got, _ := json.Marshal(rs[1].Body["messages"]); string(got) != `[{"content":"hi","role":"user"}]` {
		t.Errorf("second request %s", got)
	}
}
