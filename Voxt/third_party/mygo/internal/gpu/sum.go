//go:build !js

package gpu

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

// Browsers compile their shaders as they load (internal/gpu/gl), none ahead
// of time: builds for them leave SourceSum out, with crypto/sha256.

// SourceSum returns the SHA-256, in hex, of a shader's source as the
// repository holds it, with LF line endings, which shaders compiled ahead
// of time record: the source matches them however it was checked out, as
// with CRLF endings on Windows.
func SourceSum(src string) string {
	sum := sha256.Sum256([]byte(strings.ReplaceAll(src, "\r\n", "\n")))
	return hex.EncodeToString(sum[:])
}
