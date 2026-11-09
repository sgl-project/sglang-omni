package llm

import (
	"maps"
	"strings"
)

// APIResponses is Preset.API for OpenAI's Responses API
// (POST {base}/responses).
const APIResponses = "responses"

// minOutputTokens is the smallest max_output_tokens the Responses API
// accepts (its reference: "minimum: 16"); Probe asks for 8.
const minOutputTokens = 16

func (o *OpenAI) responsesAPI() bool { return o.preset.API == APIResponses }

func (o *OpenAI) path() string {
	if o.responsesAPI() {
		return "/responses"
	}
	return "/chat/completions"
}

// needsResponses reports whether a refusal says the model works only over
// the Responses API. OpenAI has answered this with 404, which HTTPError
// would report as a missing model.
func (o *OpenAI) needsResponses(status int, detail string) bool {
	return !o.responsesAPI() && (status == 400 || status == 404) && strings.Contains(strings.ToLower(detail), "v1/responses")
}

// responsesBody rewrites b, a finished chat completions body, for the
// Responses API, so the two formats share what goes into a request. A key
// of extra, the user's Extra Body JSON, wins over what is written here.
func responsesBody(b, extra map[string]any) map[string]any {
	put := func(key string, v any) {
		if _, own := extra[key]; !own {
			b[key] = v
		}
	}
	nest := func(parent, key string, v any) {
		m, _ := b[parent].(map[string]any)
		m = maps.Clone(m) // it may be the user's ExtraBody
		if m == nil {
			m = map[string]any{}
		}
		if _, own := m[key]; !own {
			m[key] = v
		}
		b[parent] = m
	}

	msgs, _ := b["messages"].([]oaMessage)
	delete(b, "messages")
	var system []string
	input := make([]oaMessage, 0, len(msgs))
	for _, m := range msgs {
		if m.Role == string(System) {
			system = append(system, m.Content)
		} else {
			input = append(input, m)
		}
	}
	put("input", input)
	if len(system) > 0 {
		put("instructions", strings.Join(system, "\n\n"))
	}
	// Responses are kept on the server unless told otherwise, and the
	// user's dictation should not be: Extra Body JSON cannot turn this on.
	b["store"] = false

	for _, key := range []string{"max_completion_tokens", "max_tokens"} {
		if v, ok := b[key]; ok {
			delete(b, key)
			if n, isInt := v.(int); isInt {
				v = max(n, minOutputTokens)
			}
			put("max_output_tokens", v)
		}
	}
	if rf, ok := b["response_format"].(map[string]any); ok {
		delete(b, "response_format")
		format := map[string]any{"type": rf["type"]}
		if schema, ok := rf["json_schema"].(map[string]any); ok {
			maps.Copy(format, schema)
		}
		nest("text", "format", format)
	}
	if v, ok := b["verbosity"]; ok {
		delete(b, "verbosity")
		nest("text", "verbosity", v)
	}
	if v, ok := b["reasoning_effort"]; ok {
		delete(b, "reasoning_effort")
		nest("reasoning", "effort", v)
	}

	// Not fields of the Responses API (logprobs needs an include this
	// transport has no use for; usage comes in response.completed).
	for _, key := range []string{"stream_options", "stop", "seed", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty", "logprobs", "top_logprobs", "keep_alive", "options"} {
		if _, own := extra[key]; !own {
			delete(b, key)
		}
	}
	return b
}
