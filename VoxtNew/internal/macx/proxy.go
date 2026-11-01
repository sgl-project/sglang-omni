package macx

// Proxies are the system's web proxies (System Settings → Network →
// Details → Proxies), which a GUI app does not get as https_proxy.
type Proxies struct {
	// HTTP and HTTPS are the proxies' "host:port", "" when off.
	HTTP, HTTPS string
	// Exceptions are the hosts and domains that bypass them ("*.local",
	// "169.254/16").
	Exceptions []string
	// ExcludeSimple: plain host names (no dot) bypass them.
	ExcludeSimple bool
}

// SystemProxies reads the system's proxies (CFNetwork); zero off macOS
// or when they cannot be read. Any goroutine; it takes a millisecond or
// so: callers cache it.
func SystemProxies() Proxies { return systemProxies() }
