//go:build !windows

package historyapi

import (
	"net"
	"os"
)

// Listen makes the socket at path, 0600, replacing one a crash left
// behind. Binding then changing the mode leaves a moment with the umask's
// mode, inside a home other accounts cannot enter.
func Listen(path string) (net.Listener, error) {
	if info, err := os.Lstat(path); err == nil && info.Mode()&os.ModeSocket != 0 {
		if err := os.Remove(path); err != nil {
			return nil, err
		}
	}
	listener, err := net.Listen("unix", path)
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(path, 0o600); err != nil {
		listener.Close()
		return nil, err
	}
	return listener, nil
}

func dial(path string) (net.Conn, error) {
	return net.Dial("unix", path)
}
