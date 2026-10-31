package models

// The family's acceptance scenarios (spec §16.1, 1–3) on a Mac, with the
// real Hugging Face cache and network (VOXTNEW_FAMILY_ACCEPT=1):
//
//	VOXTNEW_FAMILY_ACCEPT=1 go test -count=1 -timeout 1h -v -run TestAccept ./internal/models
//
// The real cache is only read (scenario 1). Deleting and downloading run
// in a scratch $HUB beside it (APFS clones of the real files: no room
// taken) and a scratch family directory, so that the user's apps' claims
// and models are never touched. The other apps are played by actnowkit
// (claims, locks, a peer downloader) under their bundle ids.

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/conformance"
	"github.com/hehehai/actnowkit/hfhub"
	fammodels "github.com/hehehai/actnowkit/models"
)

func acceptOnly(t *testing.T) {
	if os.Getenv("VOXTNEW_FAMILY_ACCEPT") != "1" || runtime.GOOS != "darwin" {
		t.Skip("set VOXTNEW_FAMILY_ACCEPT=1 on a Mac")
	}
}

// scratch is a temporary family directory with AudioSmith and Dicta
// registered (installed), VoxtNew's store over hub.
func scratch(t *testing.T, hub string) (*conformance.Fixture, *fammodels.Store) {
	fx := conformance.New(t) // sets ACTNOW_FAMILY_DIR and HF_HUB_CACHE for the test
	if hub == "" {
		hub = fx.Hub
	}
	fx.Hub = hub
	fx.Register("dev.hehehai.audiosmith")
	fx.Register("dev.hehehai.dicta")
	return fx, fx.Peer("dev.hehehai.voxtnew")
}

// dead is an endpoint nothing answers: a download would fail.
const dead = "http://127.0.0.1:9"

// TestAcceptReuse (scenario 1): the fast speech model, which AudioSmith
// and Dicta keep in the real cache at the family's commit, is installed
// for VoxtNew at once, with no endpoint reachable; "Use them" claims it,
// downloading nothing.
func TestAcceptReuse(t *testing.T) {
	acceptOnly(t)
	hub := Hub()
	_, store := scratch(t, hub)
	m := Open(Options{Store: store, Network: Network{Endpoints: []string{dead}}})
	defer m.Close()
	s := m.Status(ASRFast)
	if !s.Installed || s.Revision != "313d850181767edf09f00a9c289becca70e58cd0" {
		t.Fatalf("asr.fast in %s: %+v", hub, s)
	}
	t.Logf("asr.fast installed at %s, other apps' claims: %v", m.Dir(ASRFast), s.Users)
	for _, f := range m.Survey(ASRAccurate, ASRFast, VADSilero, LLMBalanced, LLMQwen3, LLMGemma4) {
		t.Logf("on this Mac: %s (%d MB) by %v", f.ID, f.Bytes>>20, f.By)
	}
	if err := m.Use(context.Background(), ASRFast); err != nil {
		t.Fatal(err)
	}
	if !m.Status(ASRFast).Mine {
		t.Error("not claimed")
	}
}

// TestAcceptDeleteShared (scenario 2): with AudioSmith claiming the fast
// speech model, VoxtNew's "Remove from VoxtNew" keeps its files; once
// AudioSmith deletes it too, they go. In a scratch $HUB of APFS clones.
func TestAcceptDeleteShared(t *testing.T) {
	acceptOnly(t)
	mod, _ := Lookup(ASRFast)
	hub := filepath.Join(t.TempDir(), "hub")
	os.MkdirAll(hub, 0o755)
	src := fammodels.RepoDir(Hub(), mod.Repo)
	if out, err := exec.Command("cp", "-Rc", src, hub+"/").CombinedOutput(); err != nil {
		t.Fatalf("clone %s: %v %s", src, err, out)
	}
	fx, store := scratch(t, hub)
	ctx := context.Background()
	smith := fx.Peer("dev.hehehai.audiosmith")
	if err := smith.Claim(ctx, mod.Claim("", fammodels.OriginDownloaded)); err != nil {
		t.Fatal(err)
	}
	m := Open(Options{Store: store, Network: Network{Endpoints: []string{dead}}})
	defer m.Close()
	if err := m.Use(ctx, ASRFast); err != nil {
		t.Fatal(err)
	}
	d, err := m.Decide(ASRFast)
	if err != nil || !d.InUse() {
		t.Fatalf("%+v %v", d, err)
	}
	t.Logf("delete asks: used by %v, frees %d MB if deleted for all apps", d.UserNames(), d.FreeBytes>>20)
	res, err := m.Delete(ctx, ASRFast, fammodels.ThisApp)
	if err != nil || len(res) != 1 || res[0].Deleted || Installed(hub, mod) == "" {
		t.Fatalf("remove from VoxtNew: %+v %v", res, err)
	}
	t.Log("removed from VoxtNew: files kept for AudioSmith")
	r, err := smith.Delete(ctx, mod.Ref(""), fammodels.ThisApp)
	if err != nil || !r.Deleted || Installed(hub, mod) != "" {
		t.Fatalf("AudioSmith's delete: %+v %v", r, err)
	}
	t.Logf("AudioSmith deleted it: %d MB freed", r.Freed>>20)
}

// TestAcceptOneStream (scenario 3): AudioSmith (actnowkit's downloader)
// and VoxtNew download the fast speech model at once into a scratch $HUB,
// both through a counting proxy in front of the system's: one copy's
// bytes cross it, both claim the model, nothing partial is left.
func TestAcceptOneStream(t *testing.T) {
	acceptOnly(t)
	upstream := os.Getenv("VOXTNEW_ACCEPT_PROXY") // the network's way out (mac-air: 127.0.0.1:7897)
	var bytes atomic.Int64
	proxy := countingProxy(t, upstream, &bytes)
	fx, store := scratch(t, "")
	mod, _ := Lookup(ASRFast)
	client := func() *http.Client {
		return NewHTTPClient(func(*http.Request) (*url.URL, error) { return &url.URL{Scheme: "http", Host: proxy}, nil })
	}
	peer := hfhub.New(hfhub.Options{Store: fx.Peer("dev.hehehai.audiosmith"), HTTPClient: client(), ReserveBytes: -1})
	ctx := context.Background()
	peerDone := make(chan error, 1)
	go func() { peerDone <- peer.Download(ctx, mod.Entry(""), nil) }()
	time.Sleep(3 * time.Second) // the peer has planned and holds the blobs' locks
	m := Open(Options{Store: store, Network: Network{Client: client()}, ReserveBytes: -1})
	defer m.Close()
	if err := m.Download(ASRFast); err != nil {
		t.Fatal(err)
	}
	waited := ""
	for m.Status(ASRFast).Downloading {
		if w := m.Status(ASRFast).WaitingFor; w != "" && waited == "" {
			waited = w
			t.Logf("VoxtNew: waiting for %s (%d MB through the proxy so far)", w, bytes.Load()>>20)
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err := <-peerDone; err != nil {
		t.Fatalf("peer: %v", err)
	}
	s := m.Status(ASRFast)
	t.Logf("done: %d MB through the proxy for a %d MB model; VoxtNew waited for %q", bytes.Load()>>20, mod.Size()>>20, waited)
	if !s.Installed || s.Err != nil || waited != "dev.hehehai.audiosmith" {
		t.Fatalf("%+v (waited %q)", s, waited)
	}
	if n := bytes.Load(); n > mod.Size()+mod.Size()/10 {
		t.Errorf("%d bytes crossed the proxy for %d: two streams", n, mod.Size())
	}
	if !fx.Claimed("dev.hehehai.voxtnew", mod.Ref("")) || !fx.Claimed("dev.hehehai.audiosmith", mod.Ref("")) {
		t.Error("both must claim it")
	}
	if left := fx.Incomplete(mod.Repo); len(left) > 0 {
		t.Errorf("partial files: %v", left)
	}
}

// countingProxy is an HTTP CONNECT proxy on loopback counting the bytes
// it carries, through upstream (another proxy's host:port) or directly.
func countingProxy(t *testing.T, upstream string, n *atomic.Int64) string {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodConnect {
			http.Error(w, "CONNECT only", http.StatusMethodNotAllowed)
			return
		}
		var out net.Conn
		var err error
		if upstream != "" {
			out, err = net.DialTimeout("tcp", upstream, 15*time.Second)
			if err == nil {
				_, err = io.WriteString(out, "CONNECT "+r.Host+" HTTP/1.1\r\nHost: "+r.Host+"\r\n\r\n")
			}
			if err == nil {
				buf := make([]byte, 1)
				var head []byte
				for err == nil && !(len(head) >= 4 && string(head[len(head)-4:]) == "\r\n\r\n") {
					_, err = out.Read(buf)
					head = append(head, buf[0])
				}
			}
		} else {
			out, err = net.DialTimeout("tcp", r.Host, 15*time.Second)
		}
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		hj, _ := w.(http.Hijacker)
		in, _, err := hj.Hijack()
		if err != nil {
			out.Close()
			return
		}
		io.WriteString(in, "HTTP/1.1 200 Connection established\r\n\r\n")
		go func() { io.Copy(counter{out, n}, in); out.Close() }()
		io.Copy(counter{in, n}, out)
		in.Close()
	})}
	go srv.Serve(ln)
	return ln.Addr().String()
}

// counter counts what goes through to w.
type counter struct {
	w io.Writer
	n *atomic.Int64
}

func (c counter) Write(p []byte) (int, error) {
	k, err := c.w.Write(p)
	c.n.Add(int64(k))
	return k, err
}
