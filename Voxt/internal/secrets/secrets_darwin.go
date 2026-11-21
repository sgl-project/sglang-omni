//go:build darwin

package secrets

import (
	"bytes"
	"fmt"
	"os/exec"
	"strings"
)

// platformSecurity runs /usr/bin/security with args, feeding it stdin.
func platformSecurity(stdin string, args ...string) ([]byte, error) {
	cmd := exec.Command("/usr/bin/security", args...)
	cmd.Stdin = strings.NewReader(stdin)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			return out, fmt.Errorf("security: %s", msg)
		}
		return out, err
	}
	return out, nil
}
