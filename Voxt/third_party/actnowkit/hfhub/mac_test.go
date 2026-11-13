package hfhub

import (
	"os"
	"runtime"
	"testing"
)

// TestMacProxy checks what only a Mac has: the system proxy read from
// scutil. It runs with ACTNOWKIT_MAC=1 on macOS:
//
//	ACTNOWKIT_MAC=1 go test -run TestMacProxy -v ./hfhub
//
// ACTNOW_MAC_PROXY, when set, is the proxy downloads must use.
func TestMacProxy(t *testing.T) {
	if os.Getenv("ACTNOWKIT_MAC") == "" || runtime.GOOS != "darwin" {
		t.Skip("set ACTNOWKIT_MAC=1 on a Mac")
	}
	for _, v := range []string{"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"} {
		t.Setenv(v, "")
		os.Unsetenv(v)
	}
	sp := currentSystemProxy()
	auto := DescribeProxy("")
	t.Logf("system proxy (scutil): https %v, http %v, exceptions %v; downloads use %q", sp.https, sp.http, sp.exceptions, auto)
	if want := os.Getenv("ACTNOW_MAC_PROXY"); want != "" && auto != want {
		t.Errorf("automatic proxy %q, want %q", auto, want)
	}
	if DescribeProxy(ProxyDirect) != "" {
		t.Error("direct mode uses a proxy")
	}
}
