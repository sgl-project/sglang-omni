package main

// TestRealSystemProxy downloads the VAD from huggingface.co alone (no
// mirror, no proxy in the environment, the app's automatic proxy, into a
// temporary family library) on a
// Mac whose network reaches the Hub only through the system's proxy
// (mac-air: 127.0.0.1:7897), and checks every request went through it.
//
//	env -u https_proxy -u http_proxy -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u all_proxy \
//	  VOXTNEW_MODELS_REAL=1 go test -count=1 -v -run TestRealSystemProxy .

import (
	"net/http"
	"net/url"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/conformance"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
)

func TestRealSystemProxy(t *testing.T) {
	if os.Getenv("VOXTNEW_MODELS_REAL") != "1" {
		t.Skip("set VOXTNEW_MODELS_REAL=1")
	}
	for _, name := range []string{"https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "all_proxy"} {
		if os.Getenv(name) != "" {
			t.Fatalf("%s is set: run with env -u %s", name, name)
		}
	}
	system := systemProxies()
	t.Logf("system proxies: %+v", system)
	if system.HTTPS == "" {
		t.Skip("no system HTTPS proxy")
	}
	var mu sync.Mutex
	used := map[string]int{} // requests by the proxy they went through
	appProxy := models.ProxyFor("", systemProxy)
	proxy := func(request *http.Request) (*url.URL, error) {
		proxyURL, err := appProxy(request)
		mu.Lock()
		defer mu.Unlock()
		if proxyURL != nil {
			used[proxyURL.Host]++
		} else {
			used["direct"]++
		}
		return proxyURL, err
	}
	fx := conformance.New(t)
	opt := models.Options{Store: fx.Peer("dev.hehehai.voxtnew.test"),
		Network: models.Network{Endpoints: []string{"https://huggingface.co"}, Client: models.NewHTTPClient(proxy)}}
	m := models.Open(opt)
	defer m.Close()
	start := time.Now()
	if err := m.Download(models.VADSilero); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Minute)
	for m.Status(models.VADSilero).Downloading && time.Now().Before(deadline) {
		time.Sleep(100 * time.Millisecond)
	}
	s := m.Status(models.VADSilero)
	t.Logf("silero-vad: %+v in %v, requests by proxy: %v", s, time.Since(start).Round(time.Millisecond), used)
	if !s.Installed || s.Err != nil {
		t.Fatalf("not installed: %+v", s)
	}
	if used[system.HTTPS] == 0 || len(used) != 1 {
		t.Errorf("requests by proxy %v, want all through %s", used, system.HTTPS)
	}
}
