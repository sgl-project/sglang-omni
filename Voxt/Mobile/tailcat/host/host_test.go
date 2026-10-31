package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/tailscale/tailcat"
	"tailscale.com/tailcfg"
	"tailscale.com/types/key"
)

func TestEmptyAllowlistRejectedBeforeStart(t *testing.T) {
	out, started := reject(t, `{"version":1,"state":null,"allowedClients":[],"localPort":8080}`)
	if started {
		t.Fatal("empty allowlist started the server")
	}
	if strings.TrimSpace(out) != `{"error":"invalid-allowlist"}` {
		t.Fatal("empty allowlist did not return the fixed error")
	}
}

func TestInvalidKeyPortAndStateRejectedBeforeStart(t *testing.T) {
	client := mustNodeKey(t)
	cases := []struct {
		body string
		want string
		leak string
	}{
		{`{"version":2,"state":null,"allowedClients":["` + client + `"],"localPort":8080}`, "invalid-version", ""},
		{`{"version":1,"state":null,"allowedClients":["` + client + `"],"localPort":0}`, "invalid-port", ""},
		{`{"version":1,"state":null,"allowedClients":["` + client + `"],"localPort":65536}`, "invalid-port", ""},
		{`{"version":1,"state":null,"allowedClients":["` + client + `"],"localPort":1}`, "invalid-port", ""},
		{`{"version":1,"state":null,"allowedClients":["` + client + `"],"localPort":65535}`, "invalid-port", ""},
		{`{"version":1,"state":null,"allowedClients":["` + client + `","` + client + `"],"localPort":8080}`, "invalid-allowlist", ""},
		{`{"version":1,"state":null,"allowedClients":["` + client + `"],"localPort":8080,"padding":"` + strings.Repeat("x", 70000) + `"}`, "invalid-request", ""},
		{`{"version":1,"state":null,"allowedClients":["not-a-node-key"],"localPort":8080}`, "invalid-client-key", "not-a-node-key"},
		{`{"version":1,"state":null,"allowedClients":["nodekey:00"],"localPort":8080}`, "invalid-client-key", "nodekey:00"},
		{`{"version":1,"state":{"version":1},"allowedClients":["` + client + `"],"localPort":8080}`, "invalid-state", ""},
		{`{"version":1,"state":"{\"version\":2}","allowedClients":["` + client + `"],"localPort":8080}`, "invalid-state", ""},
		{`{"version":1,"state":"not-json","allowedClients":["` + client + `"],"localPort":8080}`, "invalid-state", "not-json"},
	}
	for _, tc := range cases {
		out, started := reject(t, tc.body)
		if started || strings.TrimSpace(out) != `{"error":"`+tc.want+`"}` {
			t.Fatal("invalid configuration was not rejected before start")
		}
		if tc.leak != "" && strings.Contains(out, tc.leak) {
			t.Fatal("rejection echoed input")
		}
	}
}

func TestFreshStateIsJSONStringAndRestartRestoresSecrets(t *testing.T) {
	client := mustNodeKey(t)
	region := fixtureRegion()
	var first Spec
	var second Spec
	var calls int
	start := func(_ context.Context, spec Spec) (Running, error) {
		calls++
		if calls == 1 {
			first = spec
			return &fakeServer{addr: "tc-fixture-address", region: region}, nil
		}
		second = spec
		return &fakeServer{addr: "tc-fixture-address", region: spec.Region}, nil
	}
	reply := mustReply(t, start, `{"version":1,"state":null,"allowedClients":["`+client+`"],"localPort":18080}`)
	if reply.Version != 1 || reply.Port != virtualPort || reply.Address != "tc-fixture-address" {
		t.Fatal("fresh reply did not use the virtual port and server address")
	}
	if !strings.HasPrefix(reply.State, "{") {
		t.Fatal("reply state was not a JSON string")
	}
	var saved struct {
		Version      int    `json:"version"`
		PrivateKey   string `json:"privateKey"`
		PresharedKey string `json:"presharedKey"`
	}
	if json.Unmarshal([]byte(reply.State), &saved) != nil || saved.Version != 1 {
		t.Fatal("reply state was not version 1")
	}
	if !strings.HasPrefix(saved.PrivateKey, "privkey:") || !strings.HasPrefix(saved.PresharedKey, "psk:") {
		t.Fatal("reply state omitted a required secret field")
	}
	if !first.Key.Equal(mustPrivate(t, saved.PrivateKey)) || !first.PSK.Equal(mustPSK(t, saved.PresharedKey)) {
		t.Fatal("reply state did not match the started server")
	}
	if first.Region != nil || len(first.Allowed) != 1 {
		t.Fatal("fresh start did not leave region selection to the server")
	}
	encoded, err := json.Marshal(reply.State)
	if err != nil {
		t.Fatal("state string could not be reencoded")
	}
	again := mustReply(t, start, `{"version":1,"state":`+string(encoded)+`,"allowedClients":["`+client+`"],"localPort":18080}`)
	if again.State == "" || !second.Key.Equal(first.Key) || !second.PSK.Equal(first.PSK) {
		t.Fatal("restart did not restore the server secrets")
	}
	if second.Region == nil || second.Region.RegionID != region.RegionID || len(second.Region.Nodes) != 1 || second.Region.Nodes[0].HostName != region.Nodes[0].HostName {
		t.Fatal("restart did not restore the DERP region")
	}
}

func TestVirtualPortForwardsOnlyToLoopback(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("listener setup failed")
	}
	defer ln.Close()
	got := make(chan string, 1)
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			got <- ""
			return
		}
		defer conn.Close()
		buf := make([]byte, 32)
		n, _ := conn.Read(buf)
		_, _ = conn.Write([]byte("fixture-ok"))
		got <- string(buf[:n])
	}()
	var spec Spec
	start := func(_ context.Context, incoming Spec) (Running, error) {
		spec = incoming
		return &fakeServer{addr: "tc-fixture-address", region: fixtureRegion()}, nil
	}
	client := mustNodeKey(t)
	port := ln.Addr().(*net.TCPAddr).Port
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	done := make(chan error, 1)
	go func() {
		done <- Run(inR, outW, Option{Start: start, Signals: make(chan os.Signal)})
	}()
	if _, err := io.WriteString(inW, `{"version":1,"state":null,"allowedClients":["`+client+`"],"localPort":`+strconv.Itoa(port)+`}`+"\n"); err != nil {
		t.Fatal("request write failed")
	}
	if _, err := readReply(t, outR); err != nil {
		t.Fatal("helper did not reply")
	}
	if spec.OnTCP == nil || spec.OnTCP(22) != nil || spec.OnTCP(virtualPort) == nil {
		t.Fatal("TCP handling was not limited to the virtual port")
	}
	left, right := net.Pipe()
	defer left.Close()
	go spec.OnTCP(virtualPort)(right)
	if _, err := left.Write([]byte("fixture-body")); err != nil {
		t.Fatal("virtual connection write failed")
	}
	select {
	case body := <-got:
		if body != "fixture-body" {
			t.Fatal("virtual port did not reach the loopback listener")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("virtual port did not reach the loopback listener")
	}
	_ = left.SetReadDeadline(time.Now().Add(2 * time.Second))
	buf := make([]byte, 16)
	n, err := left.Read(buf)
	if err != nil || string(buf[:n]) != "fixture-ok" {
		t.Fatal("loopback response did not return through the virtual port")
	}
	_ = inW.Close()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("helper stayed running after stdin EOF")
	}
}

func TestStdinEOFClosesServerAndForwardedStream(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("listener setup failed")
	}
	defer ln.Close()
	accepted := make(chan net.Conn, 1)
	finished := make(chan struct{})
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			close(finished)
			return
		}
		accepted <- conn
		_, _ = io.Copy(io.Discard, conn)
		close(finished)
	}()
	fake := &fakeServer{addr: "tc-fixture-address", region: fixtureRegion(), closed: make(chan struct{})}
	var spec Spec
	start := func(_ context.Context, incoming Spec) (Running, error) {
		spec = incoming
		return fake, nil
	}
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	done := make(chan error, 1)
	go func() { done <- Run(inR, outW, Option{Start: start, Signals: make(chan os.Signal)}) }()
	client := mustNodeKey(t)
	_, _ = io.WriteString(inW, `{"version":1,"state":null,"allowedClients":["`+client+`"],"localPort":`+strconv.Itoa(ln.Addr().(*net.TCPAddr).Port)+`}`+"\n")
	if _, err := readReply(t, outR); err != nil {
		t.Fatal("helper did not reply")
	}
	left, right := net.Pipe()
	defer left.Close()
	go spec.OnTCP(virtualPort)(right)
	_, _ = left.Write([]byte("x"))
	select {
	case <-accepted:
	case <-time.After(2 * time.Second):
		t.Fatal("forward did not open the loopback stream")
	}
	_ = inW.Close()
	select {
	case <-fake.closed:
	case <-time.After(2 * time.Second):
		t.Fatal("stdin EOF did not close the server")
	}
	select {
	case <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("stdin EOF left the forwarded stream open")
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal("helper returned an error after stdin EOF")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("helper stayed running after stdin EOF")
	}
}

func TestSignalCancelsUnfinishedStart(t *testing.T) {
	signals := make(chan os.Signal, 1)
	entered := make(chan struct{})
	start := func(ctx context.Context, _ Spec) (Running, error) {
		close(entered)
		<-ctx.Done()
		return nil, ctx.Err()
	}
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	done := make(chan error, 1)
	go func() { done <- Run(inR, outW, Option{Start: start, Signals: signals}) }()
	client := mustNodeKey(t)
	_, _ = io.WriteString(inW, `{"version":1,"state":null,"allowedClients":["`+client+`"],"localPort":8080}`+"\n")
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("start did not begin")
	}
	signals <- syscall.SIGTERM
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("cancelled start returned success")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("signal did not stop an unfinished start")
	}
	_ = outW.Close()
	body, _ := io.ReadAll(outR)
	if strings.Contains(string(body), `"address"`) {
		t.Fatal("unfinished start published an address")
	}
}

func reject(t *testing.T, body string) (string, bool) {
	t.Helper()
	started := false
	var out strings.Builder
	err := Run(strings.NewReader(body+"\n"), &out, Option{
		Start: func(context.Context, Spec) (Running, error) {
			started = true
			return nil, io.EOF
		},
		Signals: make(chan os.Signal),
	})
	if err == nil {
		t.Fatal("rejected request returned success")
	}
	return out.String(), started
}

func mustReply(t *testing.T, start func(context.Context, Spec) (Running, error), body string) struct {
	Version int    `json:"version"`
	Address string `json:"address"`
	Port    int    `json:"port"`
	State   string `json:"state"`
} {
	t.Helper()
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	done := make(chan error, 1)
	go func() {
		done <- Run(inR, outW, Option{Start: start, Signals: make(chan os.Signal)})
	}()
	if _, err := io.WriteString(inW, body+"\n"); err != nil {
		t.Fatal("request write failed")
	}
	var reply struct {
		Version int    `json:"version"`
		Address string `json:"address"`
		Port    int    `json:"port"`
		State   string `json:"state"`
	}
	raw, err := readReply(t, outR)
	if err != nil || json.Unmarshal(raw, &reply) != nil || strings.Count(strings.TrimSpace(string(raw)), "\n") != 0 {
		t.Fatal("stdout was not one JSON reply")
	}
	_ = inW.Close()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("helper stayed running after the request")
	}
	return reply
}

func readReply(t *testing.T, r io.Reader) ([]byte, error) {
	t.Helper()
	ch := make(chan struct {
		b   []byte
		err error
	}, 1)
	go func() {
		b, err := bufio.NewReader(r).ReadBytes('\n')
		ch <- struct {
			b   []byte
			err error
		}{b, err}
	}()
	select {
	case got := <-ch:
		return got.b, got.err
	case <-time.After(2 * time.Second):
		t.Fatal("helper did not reply")
		return nil, io.EOF
	}
}

func mustNodeKey(t *testing.T) string {
	t.Helper()
	text, err := key.NewNode().Public().MarshalText()
	if err != nil || !strings.HasPrefix(string(text), "nodekey:") {
		t.Fatal("test client key setup failed")
	}
	return string(text)
}

func mustPrivate(t *testing.T, text string) key.NodePrivate {
	t.Helper()
	var private key.NodePrivate
	if private.UnmarshalText([]byte(text)) != nil || private.IsZero() {
		t.Fatal("state private key could not be read")
	}
	return private
}

func mustPSK(t *testing.T, text string) tailcat.PresharedKey {
	t.Helper()
	var psk tailcat.PresharedKey
	if psk.UnmarshalText([]byte(text)) != nil || psk.IsZero() {
		t.Fatal("state preshared key could not be read")
	}
	return psk
}

func fixtureRegion() *tailcfg.DERPRegion {
	return &tailcfg.DERPRegion{
		RegionID: 19, RegionCode: "fix",
		Nodes: []*tailcfg.DERPNode{{
			Name: "fix", RegionID: 19, HostName: "loopback.invalid", DERPPort: 443,
		}},
	}
}

type fakeServer struct {
	addr   string
	region *tailcfg.DERPRegion
	closed chan struct{}
	once   sync.Once
}

func (f *fakeServer) Address() string             { return f.addr }
func (f *fakeServer) Region() *tailcfg.DERPRegion { return f.region }
func (f *fakeServer) Close() error {
	f.once.Do(func() {
		if f.closed != nil {
			close(f.closed)
		}
	})
	return nil
}
