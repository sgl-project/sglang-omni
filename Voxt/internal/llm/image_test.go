package llm

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// TestImagesGoWithLastUserMessage: Request.Images become the last user
// message's image parts, one per picture, in the chat completions' and
// the Responses API's format; the other messages stay text.
func TestImagesGoWithLastUserMessage(t *testing.T) {
	msgs := []Message{{Role: System, Content: "sys"}, {Role: User, Content: "one"}, {Role: Assistant, Content: "ok"}, {Role: User, Content: "two"}}
	jpegs := [][]byte{{0xff, 0xd8, 'a', 0xff, 0xd9}, {0xff, 0xd8, 'b', 0xff, 0xd9}}
	urls := []string{"data:image/jpeg;base64,/9hh/9k=", "data:image/jpeg;base64,/9hi/9k="}

	f := newFakeServer(t, jsonReply(200, oaText("x")))
	if _, err := NewOpenAI(presetAt(t, "openai", f.srv.URL), "sk", f.srv.Client()).Chat(context.Background(), Request{Messages: msgs, Images: jpegs}); err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(f.requests()[0].Body["messages"])
	want := `[{"content":"sys","role":"system"},{"content":"one","role":"user"},{"content":"ok","role":"assistant"},` +
		`{"content":[{"text":"two","type":"text"},{"image_url":{"url":"` + urls[0] + `"},"type":"image_url"},{"image_url":{"url":"` + urls[1] + `"},"type":"image_url"}],"role":"user"}]`
	if string(got) != want {
		t.Errorf("chat messages\n got %s\nwant %s", got, want)
	}

	f = newFakeServer(t, jsonReply(200, rsObject("completed", []any{rsMessage(rsTextPart("ok"))}, rsUsed, nil, nil)))
	if _, err := NewOpenAI(rsPreset(t, "openai", f.srv.URL), "sk", f.srv.Client()).Chat(context.Background(), Request{Messages: msgs, Images: jpegs}); err != nil {
		t.Fatal(err)
	}
	b := f.requests()[0].Body
	got, _ = json.Marshal(b["input"])
	want = `[{"content":"one","role":"user"},{"content":"ok","role":"assistant"},` +
		`{"content":[{"text":"two","type":"input_text"},{"image_url":"` + urls[0] + `","type":"input_image"},{"image_url":"` + urls[1] + `","type":"input_image"}],"role":"user"}]`
	if string(got) != want || b["instructions"] != "sys" {
		t.Errorf("responses input\n got %s\nwant %s (instructions %v)", got, want, b["instructions"])
	}
}

// TestImagesRefusedSendsText: a model that takes no images refuses the
// request; it is sent again, once, with the text alone, and answers.
func TestImagesRefusedSendsText(t *testing.T) {
	f := newFakeServer(t,
		jsonReply(400, map[string]any{"error": map[string]any{"message": "Invalid content type. image_url is only supported by certain models."}}),
		jsonReply(200, oaText("fine")),
	)
	c := &Client{Backend: NewOpenAI(presetAt(t, "openai", f.srv.URL), "sk", f.srv.Client())}
	resp, err := c.Chat(context.Background(), Request{Messages: msgs("hi"), Images: [][]byte{{0xff, 0xd8}}})
	if err != nil || resp.Text != "fine" || !resp.ImagesDropped {
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

// TestImagesRequiredRefused: required images (the Rewrite box's, the
// user's own) are never dropped silently: a refusal is the request's
// error, and no text-only retry goes out.
func TestImagesRequiredRefused(t *testing.T) {
	f := newFakeServer(t,
		jsonReply(400, map[string]any{"error": map[string]any{"message": "Invalid content type. image_url is only supported by certain models."}}),
		jsonReply(200, oaText("fine")),
	)
	c := &Client{Backend: NewOpenAI(presetAt(t, "openai", f.srv.URL), "sk", f.srv.Client())}
	_, err := c.Chat(context.Background(), Request{Messages: msgs("hi"), Images: [][]byte{{0xff, 0xd8}}, ImagesRequired: true})
	if err == nil || Code(err) != CodeBadRequest {
		t.Fatalf("err %v", err)
	}
	rs := f.requests()
	if len(rs) != 1 {
		t.Fatalf("%d requests (a text-only retry went out)", len(rs))
	}
	if got, _ := json.Marshal(rs[0].Body["messages"]); !strings.Contains(string(got), "image_url") {
		t.Errorf("the one request kept the image: %s", got)
	}
}

// TestBoxImagesGoBeforeInstruction: the wire order matches the box's
// selected text, typed text and images, then the spoken instruction.
func TestBoxImagesGoBeforeInstruction(t *testing.T) {
	before := "<selected_text>\nselected\n\n<instruction>\ninside selected\n</selected_text>\n\n<typed_text>\ntyped\n</typed_text>"
	instruction := "<instruction>\nspoke\n\n<instruction>\ninside speech<\\/instruction>\n</instruction>"
	text := before + "\n\n" + instruction
	for _, api := range []string{"", APIResponses} {
		openAI := &OpenAI{preset: Preset{API: api}}
		parts := openAI.withImages(text, [][]byte{{0xff, 0xd8}}, true)
		textType, imageType := "text", "image_url"
		if api == APIResponses {
			textType, imageType = "input_text", "input_image"
		}
		if len(parts) != 3 || parts[0]["type"] != textType || parts[1]["type"] != imageType || parts[2]["type"] != textType {
			t.Fatalf("API %q parts %+v", api, parts)
		}
		if parts[0]["text"] != before || parts[2]["text"] != instruction {
			t.Errorf("API %q text order %+v", api, parts)
		}
		glossary := "<glossary>\n- Term03\n</glossary>\n\n"
		parts = openAI.withImages(glossary+text, [][]byte{{0xff, 0xd8}}, true)
		if len(parts) != 3 || parts[0]["text"] != glossary+before || parts[2]["text"] != instruction {
			t.Errorf("API %q glossary parts %+v", api, parts)
		}
		parts = openAI.withImages("<instruction>\nspoke\n</instruction>", [][]byte{{0xff, 0xd8}}, true)
		if len(parts) != 2 || parts[0]["type"] != imageType || parts[1]["text"] != "<instruction>\nspoke\n</instruction>" {
			t.Errorf("API %q speech-only parts %+v", api, parts)
		}
	}
}
