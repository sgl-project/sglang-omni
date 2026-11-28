package profile

import "path/filepath"

// WriteInteractiveReport writes one private result containing all three methods.
func WriteInteractiveReport(outputDir string, report InteractiveReport) error {
	return writeProfileJSON(filepath.Join(outputDir, "result.json"), report)
}
