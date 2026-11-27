package settings

import "time"

// Network: how model downloads reach the Hub, VoxtNew's own values (used
// while it does not follow the ActNow family's, Family.Follow). Proxy is
// "" (automatic: the environment's, else the Mac's), "direct" or a proxy
// URL; HFEndpoints "" or "auto" (the Hub and hf-mirror.com raced),
// "official", "mirror", or URLs (family spec §9.1).
type Network struct {
	Proxy       string `json:"proxy,omitempty"`
	HFEndpoints string `json:"hfEndpoints,omitempty"`
	// ProxyUser is a custom proxy's user name and ProxyAuthHost the proxy
	// host it and its password (the Keychain's "proxy.<host>") are for:
	// Voxt's only, never the family's (models_proxy_auth.go).
	ProxyUser     string `json:"proxyUser,omitempty"`
	ProxyAuthHost string `json:"proxyAuthHost,omitempty"`
}

// Family is VoxtNew's part in the ActNow family (internal/actnow).
type Family struct {
	// Follow says, by shared key ("general.language", "network.proxy",
	// "network.hfEndpoints", "models.root"), whether VoxtNew uses the
	// family's value; a key that is missing follows (family spec §9.3).
	Follow map[string]bool `json:"follow,omitempty"`
	// Migrated: the first start with the family's protocol is done (the
	// models on disk claimed, the shared settings merged).
	Migrated bool `json:"migrated,omitempty"`
	// Conflicts are the shared keys the family had another value of at the
	// migration: their row says so once, until the user acts on it.
	Conflicts []string `json:"conflicts,omitempty"`
	// Prompted are the shortcut conflicts whose prompt was shown
	// (actnowkit shortcuts.Conflict.Key): each pair once.
	Prompted []string `json:"prompted,omitempty"`
	// KeysSince is when each declared shortcut got its keys (by the
	// manifest's shortcut id): the later of two conflicting apps prompts.
	KeysSince map[string]KeysSince `json:"keysSince,omitempty"`
}

// KeysSince is a shortcut's keys and when they were set.
type KeysSince struct {
	Keys  string    `json:"keys"`
	Since time.Time `json:"since"`
}

// Follows reports whether key follows the family.
func (f Family) Follows(key string) bool {
	v, ok := f.Follow[key]
	return !ok || v
}
