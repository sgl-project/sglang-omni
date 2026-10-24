package main

import (
	"context"
	"encoding/json"
	"net"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tailscale/tailcat"
	"tailscale.com/types/key"
)

func TestStartRejectsInvalidInputs(t *testing.T) {
	addr := testAddress(t)
	clientKey := testClientKey(t)
	cases := []struct {
		address string
		key     string
		port    int
		want    string
	}{
		{address: "", key: clientKey, port: 1, want: `{"error":"invalid-address"}`},
		{address: "banana", key: clientKey, port: 1, want: `{"error":"invalid-address"}`},
		{address: addr, key: "", port: 1, want: `{"error":"invalid-client-key"}`},
		{address: addr, key: "nodekey:00", port: 1, want: `{"error":"invalid-client-key"}`},
		{address: addr, key: clientKey, port: 0, want: `{"error":"invalid-port"}`},
		{address: addr, key: clientKey, port: -1, want: `{"error":"invalid-port"}`},
		{address: addr, key: clientKey, port: 65536, want: `{"error":"invalid-port"}`},
	}
	for _, tc := range cases {
		got := Start(tc.address, tc.key, tc.port)
		stopQuiet(got)
		if got != tc.want {
			t.Fatal("unexpected start rejection")
		}
		if strings.Contains(got, tc.address) && tc.address != "" {
			t.Fatal("start error included address")
		}
		if strings.Contains(got, "privkey:") || strings.Contains(got, "psk:") || strings.Contains(got, "nodekey:") {
			t.Fatal("start error included key material")
		}
	}
}

func TestTwoHandlesDialOnAcceptAndStopIndependently(t *testing.T) {
	var mu sync.Mutex
	var dialed []int
	var closed []int
	installSession(t, func(remotePort int) sessionDial {
		port := remotePort
		return sessionDial{
			dial: func(context.Context) (net.Conn, error) {
				mu.Lock()
				dialed = append(dialed, port)
				mu.Unlock()
				return nil, errZeroKey
			},
			close: func() {
				mu.Lock()
				closed = append(closed, port)
				mu.Unlock()
			},
		}
	})
	addr := testAddress(t)
	clientKey := testClientKey(t)
	first := mustStart(t, addr, clientKey, 443)
	second := mustStart(t, addr, clientKey, 444)
	if first.Handle == second.Handle || first.Port == second.Port {
		t.Fatal("handles were not independent")
	}
	if count := dialCount(&mu, &dialed); count != 0 {
		t.Fatal("dialed before accept")
	}
	conn := acceptPort(t, first.Port)
	conn.Close()
	waitDials(t, &mu, &dialed, 1)
	if remote := dialedRemote(&mu, &dialed, 0); remote != 443 {
		t.Fatal("first accept dialed the wrong remote port")
	}
	conn = acceptPort(t, second.Port)
	conn.Close()
	waitDials(t, &mu, &dialed, 2)
	if remote := dialedRemote(&mu, &dialed, 1); remote != 444 {
		t.Fatal("second accept dialed the wrong remote port")
	}
	if Stop(first.Handle) != 0 || Stop(first.Handle) != 0 {
		t.Fatal("stop was not idempotent")
	}
	if got := snapshotInts(&mu, &closed); len(got) != 1 || got[0] != 443 {
		t.Fatal("first stop did not close only its client")
	}
	refusePort(t, first.Port)
	conn = acceptPort(t, second.Port)
	conn.Close()
	waitDials(t, &mu, &dialed, 3)
	if remote := dialedRemote(&mu, &dialed, 2); remote != 444 {
		t.Fatal("stopped handle affected the other remote port")
	}
	if Stop(second.Handle) != 0 || Stop(second.Handle) != 0 {
		t.Fatal("second stop was not idempotent")
	}
	if got := snapshotInts(&mu, &closed); len(got) != 2 || got[0] != 443 || got[1] != 444 {
		t.Fatal("second stop did not close only its client")
	}
	refusePort(t, second.Port)
}

func TestCloseHookRunsOnceAfterDialCancel(t *testing.T) {
	var mu sync.Mutex
	var order []string
	entered := make(chan struct{})
	var enteredOnce sync.Once
	installSession(t, func(int) sessionDial {
		return sessionDial{
			dial: func(ctx context.Context) (net.Conn, error) {
				enteredOnce.Do(func() { close(entered) })
				<-ctx.Done()
				mu.Lock()
				order = append(order, "dial-cancel")
				mu.Unlock()
				return nil, ctx.Err()
			},
			close: func() {
				mu.Lock()
				order = append(order, "hook")
				mu.Unlock()
			},
		}
	})
	started := mustStart(t, testAddress(t), testClientKey(t), 443)
	conn := acceptPort(t, started.Port)
	defer conn.Close()
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("accept did not start the dial")
	}
	if Stop(started.Handle) != 0 || Stop(started.Handle) != 0 {
		t.Fatal("stop was not idempotent")
	}
	if got := snapshotStrings(&mu, &order); len(got) != 2 || got[0] != "dial-cancel" || got[1] != "hook" {
		t.Fatal("close hook did not run once after the dial was canceled")
	}
}

func TestBindAndPortFailureCloseHookOnce(t *testing.T) {
	t.Run("bind", func(t *testing.T) {
		hooks := 0
		installSession(t, func(int) sessionDial {
			return sessionDial{
				dial:  func(context.Context) (net.Conn, error) { return nil, errZeroKey },
				close: func() { hooks++ },
			}
		})
		openForward = func(func(context.Context) (net.Conn, error)) (*Forward, error) {
			return nil, errZeroKey
		}
		got := Start(testAddress(t), testClientKey(t), 443)
		if got != `{"error":"bind-failed"}` || hooks != 1 {
			t.Fatal("bind failure did not close the client once")
		}
	})
	t.Run("port", func(t *testing.T) {
		hooks := 0
		var listener net.Listener
		installSession(t, func(int) sessionDial {
			return sessionDial{
				dial:  func(context.Context) (net.Conn, error) { return nil, errZeroKey },
				close: func() { hooks++ },
			}
		})
		openForward = func(func(context.Context) (net.Conn, error)) (*Forward, error) {
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				return nil, err
			}
			listener = ln
			ctx, cancel := context.WithCancel(context.Background())
			return &Forward{
				listener: badHostListener{Listener: ln},
				ctx:      ctx,
				cancel:   cancel,
				active:   map[net.Conn]bool{},
			}, nil
		}
		got := Start(testAddress(t), testClientKey(t), 443)
		if got != `{"error":"bind-failed"}` || hooks != 1 || listener == nil {
			t.Fatal("port failure did not close the client once")
		}
		conn, err := net.DialTimeout("tcp", listener.Addr().String(), 200*time.Millisecond)
		if err == nil {
			conn.Close()
			t.Fatal("port failure left the listener open")
		}
	})
}

func TestStopClosesOutsideRegistryLock(t *testing.T) {
	release := make(chan struct{})
	var releaseOnce sync.Once
	finish := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(finish)
	entered := make(chan struct{})
	var once sync.Once
	installDial(t, func(int) func(context.Context) (net.Conn, error) {
		return func(context.Context) (net.Conn, error) {
			once.Do(func() { close(entered) })
			<-release
			return nil, errZeroKey
		}
	})
	addr := testAddress(t)
	clientKey := testClientKey(t)
	first := mustStart(t, addr, clientKey, 443)
	conn := acceptPort(t, first.Port)
	defer conn.Close()
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("dial was not started by accept")
	}
	stopped := make(chan int, 1)
	go func() { stopped <- Stop(first.Handle) }()
	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	if _, err := conn.Read(make([]byte, 1)); err == nil {
		finish()
		t.Fatal("stop did not close the active stream")
	}
	started := make(chan int64, 1)
	go func() {
		second := mustStart(t, addr, clientKey, 444)
		started <- second.Handle
	}()
	var second int64
	select {
	case second = <-started:
	case <-time.After(2 * time.Second):
		finish()
		t.Fatal("second start blocked behind stop")
	}
	select {
	case <-stopped:
		finish()
		Stop(second)
		t.Fatal("stop finished while dial was still blocked")
	default:
	}
	finish()
	select {
	case code := <-stopped:
		if code != 0 {
			t.Fatal("blocked stop failed")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("stop did not finish after dial returned")
	}
	if Stop(first.Handle) != 0 || Stop(second) != 0 {
		t.Fatal("stop after close was not idempotent")
	}
}

func installDial(t *testing.T, dial func(remotePort int) func(context.Context) (net.Conn, error)) {
	t.Helper()
	installSession(t, func(remotePort int) sessionDial {
		return sessionDial{dial: dial(remotePort), close: func() {}}
	})
}

func installSession(t *testing.T, makeDial func(remotePort int) sessionDial) {
	t.Helper()
	prevDial := newSessionDial
	prevOpen := openForward
	newSessionDial = func(_ string, _ key.NodePrivate, remotePort int) sessionDial {
		return makeDial(remotePort)
	}
	t.Cleanup(func() {
		newSessionDial = prevDial
		openForward = prevOpen
	})
}

type badHostListener struct{ net.Listener }

func (badHostListener) Addr() net.Addr { return staticAddr("192.0.2.1:9") }

type staticAddr string

func (staticAddr) Network() string  { return "tcp" }
func (a staticAddr) String() string { return string(a) }

func snapshotInts(mu *sync.Mutex, values *[]int) []int {
	mu.Lock()
	defer mu.Unlock()
	out := make([]int, len(*values))
	copy(out, *values)
	return out
}

func snapshotStrings(mu *sync.Mutex, values *[]string) []string {
	mu.Lock()
	defer mu.Unlock()
	out := make([]string, len(*values))
	copy(out, *values)
	return out
}

func testAddress(t *testing.T) string {
	t.Helper()
	private := tailcat.NewPrivateKey()
	private.Public.RegionID = 10
	addr := string(private.Public.Addr())
	if !strings.HasPrefix(addr, "tc") {
		t.Fatal("test address setup failed")
	}
	return addr
}

func testClientKey(t *testing.T) string {
	t.Helper()
	text, err := key.NewNode().MarshalText()
	if err != nil || !strings.HasPrefix(string(text), "privkey:") {
		t.Fatal("test client key setup failed")
	}
	return string(text)
}

type started struct {
	Handle int64
	Port   int
}

func mustStart(t *testing.T, address, clientKey string, remotePort int) started {
	t.Helper()
	raw := Start(address, clientKey, remotePort)
	var doc struct {
		Handle int64  `json:"handle"`
		Port   int    `json:"port"`
		Error  string `json:"error"`
	}
	if err := json.Unmarshal([]byte(raw), &doc); err != nil || doc.Error != "" || doc.Handle == 0 || doc.Port == 0 {
		stopQuiet(raw)
		t.Fatal("start failed")
	}
	t.Cleanup(func() { Stop(doc.Handle) })
	return started{Handle: doc.Handle, Port: doc.Port}
}

func stopQuiet(raw string) {
	var doc struct {
		Handle int64 `json:"handle"`
	}
	if json.Unmarshal([]byte(raw), &doc) == nil && doc.Handle != 0 {
		Stop(doc.Handle)
	}
}

func acceptPort(t *testing.T, port int) net.Conn {
	t.Helper()
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), time.Second)
	if err != nil {
		t.Fatal("expected accept")
	}
	return conn
}

func refusePort(t *testing.T, port int) {
	t.Helper()
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), 200*time.Millisecond)
	if err == nil {
		conn.Close()
		t.Fatal("expected closed listener")
	}
}

func dialCount(mu *sync.Mutex, dialed *[]int) int {
	mu.Lock()
	defer mu.Unlock()
	return len(*dialed)
}

func dialedRemote(mu *sync.Mutex, dialed *[]int, index int) int {
	mu.Lock()
	defer mu.Unlock()
	return (*dialed)[index]
}

func waitDials(t *testing.T, mu *sync.Mutex, dialed *[]int, n int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if dialCount(mu, dialed) >= n {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("accept did not dial")
}
