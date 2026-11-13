package hfhub

import (
	"bufio"
	"net"
	"net/http"
	"net/url"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Proxies. Many Macs reach the internet only through the proxy set in
// System Settings → Network (a local VPN client's 127.0.0.1:7897, say).
// URLSession follows it; Go's HTTP client does not, so the Client asks
// for it. Options.Proxy is one of:
//
//	""         automatic: $HTTPS_PROXY / $HTTP_PROXY / $ALL_PROXY (as Go reads
//	           them, honouring $NO_PROXY), else the macOS system proxy
//	"direct"   no proxy
//	a URL      that proxy: http://host:port, https://…, socks5://…
//
// Loopback hosts never go through a proxy.
const ProxyDirect = "direct"

// systemProxy is the macOS proxy configuration (scutil --proxy).
type systemProxy struct {
	http, https, socks *url.URL
	exceptions         []string
	simpleHosts        bool // ExcludeSimpleHostnames
}

var sysProxy struct {
	sync.Mutex
	cfg  systemProxy
	read time.Time
}

// scutilProxy runs scutil --proxy; swapped in tests.
var scutilProxy = func() (string, error) {
	if runtime.GOOS != "darwin" {
		return "", nil
	}
	out, err := exec.Command("/usr/sbin/scutil", "--proxy").Output()
	return string(out), err
}

// currentSystemProxy returns the system proxy, read at most every 30 s.
func currentSystemProxy() systemProxy {
	sysProxy.Lock()
	defer sysProxy.Unlock()
	if time.Since(sysProxy.read) > 30*time.Second {
		out, err := scutilProxy()
		if err == nil {
			sysProxy.cfg = parseScutil(out)
		}
		sysProxy.read = time.Now()
	}
	return sysProxy.cfg
}

// parseScutil reads the dictionary scutil --proxy prints:
//
//	<dictionary> {
//	  ExceptionsList : <array> {
//	    0 : *.local
//	  }
//	  HTTPEnable : 1
//	  HTTPPort : 7897
//	  HTTPProxy : 127.0.0.1
//	  …
//	}
func parseScutil(out string) systemProxy {
	kv := map[string]string{}
	var exceptions []string
	inExceptions := false
	sc := bufio.NewScanner(strings.NewReader(out))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if inExceptions {
			if line == "}" {
				inExceptions = false
				continue
			}
			if _, v, ok := strings.Cut(line, " : "); ok {
				exceptions = append(exceptions, strings.TrimSpace(v))
			}
			continue
		}
		k, v, ok := strings.Cut(line, " : ")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		if k == "ExceptionsList" && strings.HasPrefix(v, "<array>") {
			inExceptions = true
			continue
		}
		kv[k] = v
	}
	proxy := func(prefix, scheme string) *url.URL {
		if kv[prefix+"Enable"] != "1" || kv[prefix+"Proxy"] == "" {
			return nil
		}
		host := kv[prefix+"Proxy"]
		if port := kv[prefix+"Port"]; port != "" {
			host = net.JoinHostPort(host, port)
		}
		return &url.URL{Scheme: scheme, Host: host}
	}
	return systemProxy{
		http:        proxy("HTTP", "http"),
		https:       proxy("HTTPS", "http"),
		socks:       proxy("SOCKS", "socks5"),
		exceptions:  exceptions,
		simpleHosts: kv["ExcludeSimpleHostnames"] == "1",
	}
}

// forURL picks the proxy for u, nil for none.
func (p systemProxy) forURL(u *url.URL) *url.URL {
	host := u.Hostname()
	if p.simpleHosts && !strings.Contains(host, ".") {
		return nil
	}
	for _, ex := range p.exceptions {
		if matchException(host, ex) {
			return nil
		}
	}
	if u.Scheme == "https" && p.https != nil {
		return p.https
	}
	if u.Scheme == "http" && p.http != nil {
		return p.http
	}
	return p.socks
}

// matchException matches host against one of macOS's bypass entries:
// "*.local", "example.com", "169.254/16", "10.0.0.0/8", "127.0.0.1".
func matchException(host, ex string) bool {
	ex = strings.TrimSpace(ex)
	if ex == "" {
		return false
	}
	if strings.Contains(ex, "/") {
		ip := net.ParseIP(host)
		if ip == nil {
			return false
		}
		addr, bits, _ := strings.Cut(ex, "/")
		for strings.Count(addr, ".") < 3 && !strings.Contains(addr, ":") {
			addr += ".0"
		}
		_, n, err := net.ParseCIDR(addr + "/" + bits)
		return err == nil && n.Contains(ip)
	}
	ex = strings.TrimPrefix(ex, "*")
	if strings.HasPrefix(ex, ".") {
		return strings.HasSuffix(host, ex) || host == ex[1:]
	}
	return strings.EqualFold(host, ex)
}

func isLoopback(host string) bool {
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// proxyFunc is the Transport's Proxy for a mode (see above).
func proxyFunc(mode string) func(*http.Request) (*url.URL, error) {
	mode = strings.TrimSpace(mode)
	var fixed *url.URL
	if mode != "" && mode != ProxyDirect {
		u, err := url.Parse(mode)
		if err != nil || u.Host == "" {
			u, err = url.Parse("http://" + mode) // "host:port"
		}
		if err == nil && u.Host != "" {
			fixed = u
		}
	}
	return func(req *http.Request) (*url.URL, error) {
		if isLoopback(req.URL.Hostname()) || mode == ProxyDirect {
			return nil, nil
		}
		if fixed != nil {
			return fixed, nil
		}
		if u, err := http.ProxyFromEnvironment(req); u != nil || err != nil {
			return u, err
		}
		return currentSystemProxy().forURL(req.URL), nil
	}
}

// ProxyFunc is the proxy rule of mode ("" automatic: environment, then the
// macOS system proxy; ProxyDirect; or a proxy URL) for an http.Transport.
// Loopback hosts never go through a proxy. Apps may use it (or
// NewHTTPClient) for their other requests (LLM, translation endpoints) so
// that one proxy setting applies to all of them.
func ProxyFunc(mode string) func(*http.Request) (*url.URL, error) { return proxyFunc(mode) }

// NewHTTPClient is an HTTP client through ProxyFunc(mode), with dial and
// TLS timeouts. responseHeader bounds the wait for response headers (the
// hub uses 30 s); 0 means no bound, for slow streaming answers such as an
// LLM's. Requests are otherwise bounded by their context.
func NewHTTPClient(mode string, responseHeader time.Duration) *http.Client {
	return &http.Client{Transport: &http.Transport{
		Proxy:                 proxyFunc(mode),
		DialContext:           (&net.Dialer{Timeout: 15 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		TLSHandshakeTimeout:   15 * time.Second,
		ResponseHeaderTimeout: responseHeader,
		MaxIdleConnsPerHost:   16,
		ForceAttemptHTTP2:     true,
	}}
}

// DescribeProxy says which proxy downloads from huggingface.co use under
// mode, for the settings page: "http://127.0.0.1:7897", or "" for none.
// It may run scutil: not from a build pass.
func DescribeProxy(mode string) string {
	req, _ := http.NewRequest(http.MethodGet, "https://huggingface.co/", nil)
	u, _ := proxyFunc(mode)(req)
	if u == nil {
		return ""
	}
	return u.Redacted()
}
