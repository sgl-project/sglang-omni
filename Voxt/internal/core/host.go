package core

import (
	"net"
	"strings"
)

// IsLoopback reports whether host (a URL's Hostname, or a bare host) is
// this Mac: localhost, a *.localhost name or a loopback address. A key
// goes over plain http or ws only to such a host.
func IsLoopback(host string) bool {
	host = strings.Trim(strings.ToLower(host), "[]")
	if host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
