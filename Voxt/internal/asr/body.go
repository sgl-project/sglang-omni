package asr

import (
	"encoding/json"
	"strings"
)

// ErrorBody pulls a code and a message out of the usual error bodies:
// {"error":{"message":…,"code":…,"type":…}}, {"error":"…"},
// {"code":…,"message":…}, {"header":{"code":…,"message":…}} (DashScope's
// task events), else the text itself.
func ErrorBody(raw []byte) (code, msg string) {
	s := strings.TrimSpace(string(raw))
	if s == "" {
		return "", ""
	}
	var v map[string]json.RawMessage
	if json.Unmarshal(raw, &v) != nil {
		return "", Clip(s, 300)
	}
	str := func(r json.RawMessage) string {
		var x any
		if json.Unmarshal(r, &x) != nil {
			return ""
		}
		switch t := x.(type) {
		case string:
			return t
		case float64:
			b, _ := json.Marshal(t)
			return string(b)
		}
		return ""
	}
	if e, ok := v["error"]; ok {
		if m := str(e); m != "" {
			return "", m
		}
		var o map[string]json.RawMessage
		if json.Unmarshal(e, &o) == nil {
			code = str(o["code"])
			if code == "" {
				code = str(o["type"])
			}
			return code, str(o["message"])
		}
	}
	if h, ok := v["header"]; ok {
		var o map[string]json.RawMessage
		if json.Unmarshal(h, &o) == nil {
			if c := str(o["error_code"]); c != "" {
				return c, str(o["error_message"])
			}
		}
	}
	for _, k := range []string{"code", "error_code", "status_code"} {
		if c := str(v[k]); c != "" {
			code = c
			break
		}
	}
	for _, k := range []string{"message", "msg", "error_message", "detail"} {
		if m := str(v[k]); m != "" {
			return code, m
		}
	}
	return code, Clip(s, 300)
}

func lower(s string) string { return strings.ToLower(s) }
