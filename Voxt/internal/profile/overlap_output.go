package profile

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// WriteOverlapReports writes one private result.json per scheduling method.
func WriteOverlapReports(outputDir string, reports []OverlapReport) error {
	var errs []error
	for _, report := range reports {
		switch report.Method {
		case "one_shot", "serial", "overlap":
		default:
			errs = append(errs, fmt.Errorf("overlap: unsafe method %q", report.Method))
			continue
		}
		if err := writeOverlapReport(filepath.Join(outputDir, report.Method, "result.json"), report); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func writeOverlapReport(path string, report OverlapReport) error {
	return writeProfileJSON(path, report)
}

func writeProfileJSON(path string, report any) (returnErr error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	var encoded bytes.Buffer
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(report); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".overlap-*.json")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer func() {
		if returnErr != nil {
			_ = os.Remove(temporaryPath)
		}
	}()
	if err := temporary.Chmod(0o600); err != nil {
		_ = temporary.Close()
		return err
	}
	if _, err := temporary.Write(encoded.Bytes()); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(temporaryPath, path)
}
