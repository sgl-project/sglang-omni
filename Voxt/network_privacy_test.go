package main

import (
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// dialedHosts records the host of every HTTP request the app sends: the
// ones through its own clients (networkFor asks systemProxy for each) and
// the ones through http.DefaultTransport.
type dialedHosts struct {
	mu    sync.Mutex
	hosts []string
	next  http.RoundTripper
}

func (d *dialedHosts) add(r *http.Request) {
	d.mu.Lock()
	d.hosts = append(d.hosts, r.URL.Host)
	d.mu.Unlock()
}

func (d *dialedHosts) RoundTrip(r *http.Request) (*http.Response, error) {
	d.add(r)
	return d.next.RoundTrip(r)
}

func (d *dialedHosts) seen() []string {
	d.mu.Lock()
	defer d.mu.Unlock()
	return append([]string(nil), d.hosts...)
}

func recordDialedHosts(t *testing.T) *dialedHosts {
	d := &dialedHosts{next: http.DefaultTransport}
	oldProxy, oldTransport := systemProxy, http.DefaultTransport
	systemProxy = func(r *http.Request) (*url.URL, error) { d.add(r); return nil, nil }
	http.DefaultTransport = d
	t.Cleanup(func() { systemProxy, http.DefaultTransport = oldProxy, oldTransport })
	return d
}

// TestDefaultSessionContactsNoRemoteHost: a session on the local models,
// with no remote model set up, sends no request off this Mac; with a
// remote language model set up, the only host it contacts is that one.
func TestDefaultSessionContactsNoRemoteHost(t *testing.T) {
	t.Run("local only", func(t *testing.T) {
		dialed := recordDialedHosts(t)
		set, keys, _ := fileSetup(t, "hello aura link", &fakeChat{text: "Hello, VoxtNew."})
		ev := script(t,
			hotkeyLine(0, "transcribe/0", proto.PhaseTrigger),
			hotkeyLine(400, "transcribe/0", proto.PhaseTrigger),
		)
		a := newSessionApp(t, ev, set, keys)
		a.pump("the end", func() bool { return a.sess.State.Ended() })
		if a.sess.State != session.Done || a.sess.Result != "Hello, VoxtNew." {
			t.Fatalf("end %+v", a.sess)
		}
		for _, host := range dialed.seen() {
			name, _, err := net.SplitHostPort(host)
			if err != nil {
				name = host
			}
			if ip := net.ParseIP(name); name != "localhost" && (ip == nil || !ip.IsLoopback()) {
				t.Errorf("the local session contacted %s", host)
			}
		}
	})
	t.Run("remote language model", func(t *testing.T) {
		dialed := recordDialedHosts(t)
		srv, _ := mockOpenAI(t, "早上好，团队。")
		useKeys(t, &memKeys{m: map[string]string{"llm." + llm.Custom: "sk-mock-123"}})
		ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger))
		a := newSessionApp(t, helpertest.Opt("selection", "Good morning, team.")+","+ev, func(s *settings.Settings) {
			s.Features.Translate.Target = "zh-Hans"
			s.Features.Translate.LLM = llm.Custom
			s.Providers.LLM = []settings.Provider{{ID: llm.Custom, Endpoint: srv.URL + "/v1", Model: "mock-model"}}
		}, local.NotInstalled{})
		a.pump("the end", func() bool { return a.sess.State.Ended() })
		if a.sess.State != session.Done {
			t.Fatalf("end %+v", a.sess)
		}
		want := strings.TrimPrefix(srv.URL, "http://")
		got := dialed.seen()
		if len(got) == 0 {
			t.Fatal("the recorder saw no request to the remote model")
		}
		for _, host := range got {
			if host != want {
				t.Errorf("contacted %s, want only %s", host, want)
			}
		}
	})
}
