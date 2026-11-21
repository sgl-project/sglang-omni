package profile

import (
	"encoding/json"
	"fmt"
	"strings"
)

func validateCorrectionResponse(raw string) (string, error) {
	candidate := strings.TrimSpace(raw)
	switch {
	case candidate == "":
		return "empty_response", fmt.Errorf("profile: correction response must be a JSON array of {i,text} objects")
	case strings.HasPrefix(candidate, "["):
		end := strings.LastIndex(candidate, "]")
		if end < 0 {
			return "malformed", fmt.Errorf("profile: correction response must be a JSON array of {i,text} objects")
		}
		candidate = candidate[:end+1]
	case strings.HasPrefix(candidate, "```"):
		lines := strings.Split(candidate, "\n")
		if len(lines) < 2 {
			return "malformed", fmt.Errorf("profile: correction response must be a JSON array of {i,text} objects")
		}
		candidate = strings.Join(lines[1:], "\n")
		if end := strings.LastIndex(candidate, "```"); end >= 0 {
			candidate = candidate[:end]
		}
		candidate = strings.TrimSpace(candidate)
	default:
		start := strings.Index(candidate, "[")
		end := strings.LastIndex(candidate, "]")
		if start < 0 || end < start {
			return "malformed", fmt.Errorf("profile: correction response must be a JSON array of {i,text} objects")
		}
		candidate = candidate[start : end+1]
	}
	var lines []struct {
		Index *int    `json:"i"`
		Text  *string `json:"text"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(candidate)), &lines); err != nil || lines == nil {
		return "malformed", fmt.Errorf("profile: correction response must be a JSON array of {i,text} objects")
	}
	for _, line := range lines {
		if line.Index == nil || line.Text == nil {
			return "malformed", fmt.Errorf("profile: correction response contains an item without i and text")
		}
	}
	if len(lines) == 0 {
		return "valid_noop", nil
	}
	return "successful_edit", nil
}
