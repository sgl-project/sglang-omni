package hfhub

import (
	"net/http"
	"net/url"
	"testing"
	"time"
)

func TestSystemProxy(t *testing.T) {
	out := `<dictionary> {
  ExceptionsList : <array> {
    0 : *.local
    1 : 169.254/16
    2 : internal.example.com
  }
  ExcludeSimpleHostnames : 1
  FTPPassive : 1
  HTTPEnable : 1
  HTTPPort : 7897
  HTTPProxy : 127.0.0.1
  HTTPSEnable : 1
  HTTPSPort : 7898
  HTTPSProxy : 127.0.0.1
  SOCKSEnable : 1
  SOCKSPort : 7899
  SOCKSProxy : 127.0.0.1
}
`
	p := parseScutil(out)
	u := func(s string) *url.URL { x, _ := url.Parse(s); return x }
	for in, want := range map[string]string{
		"https://huggingface.co/api":   "http://127.0.0.1:7898",
		"http://huggingface.co/":       "http://127.0.0.1:7897",
		"ftp://example.com/":           "socks5://127.0.0.1:7899",
		"https://printer.local/":       "",
		"https://169.254.3.4/":         "",
		"https://intranet/":            "",
		"https://internal.example.com": "",
	} {
		got := ""
		if x := p.forURL(u(in)); x != nil {
			got = x.String()
		}
		if got != want {
			t.Errorf("%s: %q, want %q", in, got, want)
		}
	}
	if p := parseScutil("<dictionary> {\n  HTTPEnable : 0\n}\n"); p.forURL(u("https://huggingface.co")) != nil {
		t.Error("disabled proxy used")
	}

	// Modes.
	saved := scutilProxy
	defer func() { scutilProxy = saved; sysProxy.read = time.Time{} }()
	scutilProxy = func() (string, error) { return out, nil }
	sysProxy.Lock()
	sysProxy.read = time.Time{}
	sysProxy.Unlock()
	for _, k := range []string{"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"} {
		t.Setenv(k, "")
	}
	req, _ := http.NewRequest("GET", "https://huggingface.co/api", nil)
	check := func(mode, want string) {
		t.Helper()
		got, err := proxyFunc(mode)(req)
		s := ""
		if got != nil {
			s = got.String()
		}
		if err != nil || s != want {
			t.Errorf("mode %q: %q %v, want %q", mode, s, err, want)
		}
	}
	check("", "http://127.0.0.1:7898")
	check("direct", "")
	check("socks5://10.0.0.2:1080", "socks5://10.0.0.2:1080")
	check("10.0.0.2:3128", "http://10.0.0.2:3128")
	local, _ := http.NewRequest("GET", "http://127.0.0.1:8080/api", nil)
	if got, _ := proxyFunc("http://10.0.0.2:3128")(local); got != nil {
		t.Error("loopback through a proxy")
	}
	if DescribeProxy("") != "http://127.0.0.1:7898" || DescribeProxy("direct") != "" {
		t.Error(DescribeProxy(""))
	}
}

// NewHTTPClient follows the proxy mode, and 0 leaves response headers unbounded.
func TestNewHTTPClient(t *testing.T) {
	c := NewHTTPClient("http://127.0.0.1:9", 0)
	tr := c.Transport.(*http.Transport)
	if tr.ResponseHeaderTimeout != 0 {
		t.Error("bounded")
	}
	req, _ := http.NewRequest("GET", "https://api.openai.com/v1/models", nil)
	if u, _ := tr.Proxy(req); u == nil || u.Host != "127.0.0.1:9" {
		t.Errorf("proxy %v", u)
	}
	local, _ := http.NewRequest("GET", "http://localhost:11434/api/tags", nil)
	if u, _ := ProxyFunc("http://127.0.0.1:9")(local); u != nil {
		t.Errorf("loopback proxied via %v", u)
	}
}
