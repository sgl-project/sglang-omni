package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"io/fs"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	famsettings "github.com/hehehai/actnowkit/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// familyValue is the family's value of a shared key ("" unset).
func familyValue(t *testing.T, a *app, key string) string {
	t.Helper()
	s, _, err := famsettings.Read(a.fam.Dir())
	if err != nil {
		t.Fatal(err)
	}
	v, _ := s.Get(key)
	return v
}

// TestNetworkDialog: the deep link opens it; a change applies to the
// family while VoxtNew follows it, to VoxtNew alone after "Use a different
// setting for this app"; a custom proxy is checked.
func TestNetworkDialog(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.famStarted = true // familySettingsChanged writes the shared settings
	addPeer(t, a, dictaID, nil)
	a.familyFound(a.fam.Discover())
	a.openURL("voxtnew://models/network")
	if a.currentPage() != "models" || !a.netOpen || !has(tt, "How model downloads reach Hugging Face") {
		t.Fatalf("deep link: page %q open %v %q", a.currentPage(), a.netOpen, tt.Texts())
	}
	if !has(tt, "Applies to all apps that use this model library") || !has(tt, "The proxy in this Mac's network settings, or one set by an environment variable.") {
		t.Fatalf("defaults: %q", tt.Texts())
	}

	// A custom proxy: checked, then the family's.
	click(t, tt, "Custom")
	a.netForm.proxyURL = "ftp://127.0.0.1:21"
	click(t, tt, "Save")
	if !a.netOpen || !has(tt, "Enter a proxy address that starts with http://, https:// or socks5://, such as http://127.0.0.1:7890.") {
		t.Fatalf("an ftp proxy was taken: %q", tt.Texts())
	}
	a.netForm.proxyURL = " socks5://127.0.0.1:1080 "
	click(t, tt, "hf-mirror.com only")
	click(t, tt, "Save")
	if a.netOpen || a.prefs.Network.Proxy != "socks5://127.0.0.1:1080" || a.prefs.Network.HFEndpoints != famsettings.EndpointsMirror {
		t.Fatalf("saved: open %v %+v", a.netOpen, a.prefs.Network)
	}
	if got := familyValue(t, a, actnow.KeyProxy); got != "socks5://127.0.0.1:1080" {
		t.Errorf("family's proxy %q", got)
	}
	if got := familyValue(t, a, actnow.KeyHFEndpoints); got != famsettings.EndpointsMirror {
		t.Errorf("family's source %q", got)
	}

	// For this app only: the family keeps its value.
	a.openNetwork()
	if a.netForm.proxyMode != proxyCustom || a.netForm.proxyURL != "socks5://127.0.0.1:1080" || a.netForm.source != 2 {
		t.Fatalf("form %+v", a.netForm)
	}
	click(t, tt, "Use a different setting for this app") // the proxy's
	if a.prefs.Family.Follows(actnow.KeyProxy) || !a.prefs.Family.Follows(actnow.KeyHFEndpoints) || !has(tt, "Only Voxt uses this setting.") {
		t.Fatalf("follow: %+v %q", a.prefs.Family.Follow, tt.Texts())
	}
	click(t, tt, "No proxy")
	click(t, tt, "Save")
	if a.prefs.Network.Proxy != famsettings.ProxyDirect || familyValue(t, a, actnow.KeyProxy) != "socks5://127.0.0.1:1080" {
		t.Errorf("own proxy %q, family's %q", a.prefs.Network.Proxy, familyValue(t, a, actnow.KeyProxy))
	}
	if got := a.fam.Effective(a.prefs, actnow.KeyProxy); got != famsettings.ProxyDirect {
		t.Errorf("effective %q", got)
	}

	// Back to the family's.
	a.openNetwork()
	click(t, tt, "Use the same setting as the other apps that share this model library")
	if !a.prefs.Family.Follows(actnow.KeyProxy) || a.prefs.Network.Proxy != "socks5://127.0.0.1:1080" || a.netForm.proxyMode != proxyCustom {
		t.Errorf("following again: %+v %+v", a.prefs.Network, a.netForm)
	}
}

// TestNetworkUseTheirs: a proxy that differed from the family's at the
// first start shows "Use theirs" once.
func TestNetworkUseTheirs(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.famStarted = true
	addPeer(t, a, audioSmithID, nil)
	a.familyFound(a.fam.Discover())
	if _, err := famsettings.Set(context.Background(), a.fam.Dir(), audioSmithID, actnow.KeyProxy, "http://10.0.0.1:8080"); err != nil {
		t.Fatal(err)
	}
	a.fam.ReadShared()
	a.updateSettings(func(s *settings.Settings) {
		s.Network.Proxy = famsettings.ProxyDirect
		s.Family.Follow = map[string]bool{actnow.KeyProxy: false}
		s.Family.Conflicts = []string{actnow.KeyProxy}
	})
	a.navigate(modelsPath)
	click(t, tt, "Network…")
	if !has(tt, "Other apps that share this model library use a different proxy.") || has(tt, "Other apps that share this model library use a different download source.") {
		t.Fatalf("no notice: %q", tt.Texts())
	}
	click(t, tt, "Use theirs")
	if !a.prefs.Family.Follows(actnow.KeyProxy) || a.prefs.Network.Proxy != "http://10.0.0.1:8080" || len(a.prefs.Family.Conflicts) != 0 {
		t.Errorf("theirs: %+v", a.prefs)
	}
	if a.netForm.proxyMode != proxyCustom || a.netForm.proxyURL != "http://10.0.0.1:8080" || !has(tt, "Applies to all apps that use this model library") {
		t.Errorf("form %+v", a.netForm)
	}

	// Shown once: closing the dialog forgets the notice.
	a.updateSettings(func(s *settings.Settings) {
		s.Family.Follow = map[string]bool{actnow.KeyHFEndpoints: false}
		s.Family.Conflicts = []string{actnow.KeyHFEndpoints}
	})
	a.openNetwork()
	if !has(tt, "Other apps that share this model library use a different download source.") {
		t.Fatalf("no notice: %q", tt.Texts())
	}
	click(t, tt, "Cancel")
	if a.netOpen || slices.Contains(a.prefs.Family.Conflicts, actnow.KeyHFEndpoints) {
		t.Errorf("still noted: %v", a.prefs.Family.Conflicts)
	}
}

// TestFamilyLinesNeedAnotherApp: the shared settings speak of the family
// only when a release of another ActNow app is installed (a dev build of
// one does not count); alone, VoxtNew says nothing of it, though the data
// still follows the family's files. Model descriptions never say "another
// app": the status line names it ("Installed by Dicta").
func TestFamilyLinesNeedAnotherApp(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.famStarted = true
	family := []string{"Applies to all apps that use this model library", "Only Voxt uses this setting.", "Use the same setting as the other apps that share this model library",
		"Use a different setting for this app"}
	// want: the network dialog's family lines; shared: Storage's, which
	// also need the other app's models in Voxt's folder.
	check := func(when string, want, shared bool) {
		t.Helper()
		a.navigate(modelsPath)
		a.openNetwork()
		frames(tt)
		if has(tt, family[0]) != want {
			t.Errorf("%s: %q shown %v", when, family[0], !want)
		}
		for _, s := range family[1:] {
			if !want && has(tt, s) {
				t.Errorf("%s: %q shown", when, s)
			}
		}
		click(t, tt, "Cancel")
		a.navigate("/settings/storage")
		frames(tt)
		if has(tt, "Shared by all apps that use this model library") != shared || has(tt, "Shared with other apps that use this model library") != shared || has(tt, "Used by Voxt") == shared {
			t.Errorf("%s: storage %q", when, tt.Texts())
		}
	}
	check("alone", false, false)
	if got := familyValue(t, a, actnow.KeyProxy); got != "" {
		t.Errorf("family's proxy %q before any change", got)
	}
	a.updateSettings(func(s *settings.Settings) { s.Network.Proxy = famsettings.ProxyDirect })
	if got := familyValue(t, a, actnow.KeyProxy); got != famsettings.ProxyDirect {
		t.Errorf("alone, a change still goes to the family: %q", got)
	}

	addPeer(t, a, dictaID+".dev", nil)
	a.familyFound(a.fam.Discover())
	check("a dev build of Dicta", false, false)

	addPeer(t, a, dictaID, nil)
	a.familyFound(a.fam.Discover())
	check("Dicta installed, its models elsewhere", true, false)
	seedModels(t, a, models.ASRFast)
	claimFor(t, a, dictaID, models.ASRFast)
	check("Dicta's models in Voxt's folder", true, true)

	for _, id := range []string{models.LLMQwen3, models.LLMGemma4} {
		if about := uitext.ModelAbout(id); about == "" || strings.Contains(strings.ToLower(about), "another") {
			t.Errorf("%s: %q", id, about)
		}
	}
}

// TestProxyAuth: a custom proxy's user name and password, as Voxt 1.x's.
// The dialog has the two fields; with both set,
// downloads, the remote services' requests and their WebSockets carry
// them to a proxy that wants them, which answers 407 without. The
// password is in the Keychain ("proxy.<host>") and nowhere on disk or in
// the logs; a new host drops the old host's item, Automatic keeps them
// for later, and a restart reads them again.
func TestProxyAuth(t *testing.T) {
	const password = "Vx-proxy-pw-90417"
	keys := &memKeys{m: map[string]string{}}
	useKeys(t, keys)
	logFile := filepath.Join(t.TempDir(), "log")
	logOut, err := os.Create(logFile)
	if err != nil {
		t.Fatal(err)
	}
	log.SetOutput(logOut)
	t.Cleanup(func() { log.SetOutput(os.Stderr); logOut.Close() })
	want := "Basic " + base64.StdEncoding.EncodeToString([]byte("alice:"+password))
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Proxy-Authorization") != want {
			w.Header().Set("Proxy-Authenticate", `Basic realm="proxy"`)
			w.WriteHeader(http.StatusProxyAuthRequired)
			return
		}
		fmt.Fprint(w, "ok "+r.URL.Host)
	}))
	t.Cleanup(proxy.Close)
	_, port, _ := net.SplitHostPort(strings.TrimPrefix(proxy.URL, "http://"))
	// statuses are the proxy's answers to a download, a remote service's
	// request and a WebSocket's handshake, through a's clients.
	statuses := func(a *app) [3]int {
		t.Helper()
		var out [3]int
		for i, client := range []*http.Client{a.networkFor(a.prefs).Client, a.remoteHTTP.Load(), asr.WSClient(a.remoteHTTP.Load())} {
			res, err := client.Get("http://service.example/v1/models")
			if err != nil {
				t.Fatalf("request %d: %v", i, err)
			}
			res.Body.Close()
			out[i] = res.StatusCode
		}
		return out
	}
	ok, refused := [3]int{200, 200, 200}, [3]int{407, 407, 407}

	dir := t.TempDir()
	a, _, tt := newTestAppIn(t, dir)
	a.famStarted = true // the proxy goes to the family's settings.json, which VoxtNew follows
	a.openURL("voxtnew://models/network")
	click(t, tt, "Custom")
	if l := tt.AccessLabels(); !slices.Contains(l, "Proxy user name") || !slices.Contains(l, "Proxy password") {
		t.Fatalf("no user name or password field: %q", l)
	}
	a.netForm.proxyURL = proxy.URL
	for _, field := range []struct{ label, text string }{{"Proxy user name", "alice"}, {"Proxy password", password}} {
		if err := tt.Click(field.label); err != nil {
			t.Fatal(err)
		}
		tt.Type(field.text)
	}
	if a.netForm.proxyUser != "alice" || a.netForm.proxyPassword != password {
		t.Fatalf("typed %+v", a.netForm)
	}
	click(t, tt, "Save")
	if keys.m["proxy.127.0.0.1"] != password || a.prefs.Network.ProxyUser != "alice" || a.prefs.Network.ProxyAuthHost != "127.0.0.1" {
		t.Fatalf("saved %v, %+v", keys.m, a.prefs.Network)
	}
	if got := statuses(a); got != ok {
		t.Fatalf("with the user: %v", got)
	}

	// Only with both: no user name, no Proxy-Authorization.
	a.openNetwork()
	a.netForm.proxyUser = ""
	click(t, tt, "Save")
	if got := statuses(a); got != refused {
		t.Errorf("without a user name: %v", got)
	}

	// Another host: the old host's item goes.
	a.openNetwork()
	if a.netForm.proxyPassword != password {
		t.Fatalf("form %+v", a.netForm)
	}
	a.netForm.proxyURL, a.netForm.proxyUser = "http://localhost:"+port, "alice"
	click(t, tt, "Save")
	if _, old := keys.m["proxy.127.0.0.1"]; old || keys.m["proxy.localhost"] != password {
		t.Fatalf("keys %v", keys.m)
	}
	if got := statuses(a); got != ok {
		t.Errorf("on the new host: %v", got)
	}

	// Automatic keeps them; Custom shows them again.
	a.openNetwork()
	click(t, tt, "Automatic")
	click(t, tt, "Save")
	a.openNetwork()
	click(t, tt, "Custom")
	if a.netForm.proxyUser != "alice" || a.netForm.proxyPassword != password || keys.m["proxy.localhost"] != password {
		t.Fatalf("after Automatic: form %+v, keys %v", a.netForm, keys.m)
	}
	a.netForm.proxyURL = "http://localhost:" + port
	click(t, tt, "Save")

	// A restart reads the password from the Keychain.
	a.quit()
	a.lib.close()
	b, _, _ := newTestAppIn(t, dir)
	if got := statuses(b); got != ok {
		t.Errorf("after a restart: %v", got)
	}

	// The error a request shows names no password: the proxy is gone.
	proxy.Close()
	if _, err := b.remoteHTTP.Load().Get("http://service.example/v1/models"); err == nil || strings.Contains(err.Error(), password) {
		t.Errorf("error %v", err)
	}
	// Nowhere on disk (the settings, the family's settings.json, the
	// database) or in the logs.
	filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			if raw, _ := os.ReadFile(path); bytes.Contains(raw, []byte(password)) {
				t.Errorf("%s has the password", path)
			}
		}
		return nil
	})
	for _, e := range b.logs.Tail(1000, "") {
		if strings.Contains(e.Message, password) {
			t.Errorf("log %q", e.Message)
		}
	}
	if raw, _ := os.ReadFile(logFile); bytes.Contains(raw, []byte(password)) || bytes.Contains(raw, []byte(want)) {
		t.Errorf("the Go log has the password: %s", raw)
	}
}
