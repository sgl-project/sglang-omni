//go:build !darwin

package macx

import "net"

func peerPID(*net.UnixConn) (int, error) { return 0, ErrUnsupported }

func procParent(int) (int, bool) { return 0, false }

func procPath(int) string { return "" }
