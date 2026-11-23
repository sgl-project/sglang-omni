//go:build !darwin

package macx

import "errors"

// ReadPDF reports that native PDF extraction needs PDFKit on macOS.
func ReadPDF(string, int) (string, [][]byte, int, error) {
	return "", nil, 0, errors.New("macx: PDF input requires macOS")
}
