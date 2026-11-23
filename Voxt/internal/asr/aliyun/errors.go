package aliyun

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// serviceError classifies an error the service reported inside the
// protocol (a task-failed event, a Realtime error event) by its code and
// message. DashScope's codes (Model Studio "error codes" reference):
//
//	InvalidApiKey, invalid_api_key, Unauthorized                → auth
//	AccessDenied (no model or billing words)                    → auth
//	Throttling* (rate limits)                                   → quota, retryable
//	Arrearage, *Quota*, FreeTierOnly, Unpurchased,
//	Model.AccessDenied, "model not activated"                   → quota
//	ModelNotFound, model_not_found, "model … not exist"         → config
//	InternalError*, SystemError, ServiceUnavailable, server_error → server
//	CLIENT_ERROR "request timeout" (no audio for ~23 s)         → network
//	anything else (InvalidParameter, BadRequest…)               → protocol
func serviceError(code, msg string) *asr.Error {
	e := &asr.Error{Provider: ID, Code: code, Detail: asr.Clip(msg, 300)}
	c := strings.ToLower(code)
	low := c + " " + strings.ToLower(msg)
	has := func(words ...string) bool {
		for _, w := range words {
			if strings.Contains(low, w) {
				return true
			}
		}
		return false
	}
	switch {
	case has("throttl", "rate limit", "ratelimit", "too many requests") && !has("arrearage", "freetieronly", "free tier"):
		e.Kind, e.Retryable = asr.KindQuota, true
	case has("arrearage", "quota", "freetieronly", "free tier", "unpurchased", "not activated", "not opened",
		"insufficient", "balance", "model.accessdenied", "access denied for model"):
		e.Kind = asr.KindQuota
	case has("invalidapikey", "invalid_api_key", "invalid api-key", "invalid api key", "unauthorized", "authentication"),
		c == "accessdenied", strings.HasPrefix(c, "accessdenied."):
		e.Kind = asr.KindAuth
	case has("modelnotfound", "model_not_found", "model not found", "model not exist", "model does not exist", "invalid model"):
		e.Kind = asr.KindConfig
	case has("internalerror", "internal_error", "internal error", "systemerror", "serviceunavailable",
		"service unavailable", "server_error", "server error", "timeout.internal"):
		e.Kind, e.Retryable = asr.KindServer, true
	case c == "client_error" && has("timeout"):
		e.Kind, e.Retryable = asr.KindNetwork, true
	case has("sample_rate", "sample rate", "audio format", "unsupported format", "invalid audio"):
		e.Kind = asr.KindAudio
	default:
		e.Kind = asr.KindProtocol
	}
	if e.Detail == "" {
		e.Detail = "the service failed the task"
	}
	return e
}

// noSpeech reports a failure that only means nothing was said: the
// stream then ends with an empty (or partial) final text, not an error.
func noSpeech(code, msg string) bool {
	return asr.NoSpeech(code, msg)
}
