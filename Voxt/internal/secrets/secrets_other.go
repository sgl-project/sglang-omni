//go:build !darwin && !windows

package secrets

// platformSecurity: no secret store on this platform.
func platformSecurity(string, ...string) ([]byte, error) {
	return nil, ErrNoKeychain
}
