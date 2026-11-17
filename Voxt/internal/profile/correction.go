package profile

import (
	"encoding/json"
	"fmt"
	"strings"
)

func validateCorrectionResponse(raw string) error {
	var lines []struct {
		Index *int    `json:"i"`
		Text  *string `json:"text"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &lines); err != nil || lines == nil {
		return fmt.Errorf("profile: correction response must be a JSON array of {i,text} objects")
	}
	for _, line := range lines {
		if line.Index == nil || line.Text == nil {
			return fmt.Errorf("profile: correction response contains an item without i and text")
		}
	}
	return nil
}
