package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"syscall"

	"github.com/tailscale/tailcat"
	"tailscale.com/tailcfg"
	"tailscale.com/types/key"
	"tailscale.com/wgengine/filter"
)

const (
	protoVersion = 1
	stateVersion = 1
	virtualPort  = 48931
)

const (
	errRequest   = "invalid-request"
	errVersion   = "invalid-version"
	errAllowlist = "invalid-allowlist"
	errClientKey = "invalid-client-key"
	errPort      = "invalid-port"
	errState     = "invalid-state"
	errStart     = "start-failed"
)

// Spec is the server a starter must bring up. Region is nil only for a fresh host.
type Spec struct {
	Port uint16 // Zero preserves the legacy normal forwarding port.
	Key     key.NodePrivate
	PSK     tailcat.PresharedKey
	Region  *tailcfg.DERPRegion
	Allowed []key.NodePublic
	OnTCP   func(uint16) func(net.Conn)
}

// Running is an owned Tailcat server. Address is the client tailcat address.
type Running interface {
	Address() string
	Region() *tailcfg.DERPRegion
	Close() error
}

// Option supplies test seams. A nil Start uses the real Tailcat server.
// A nil Signals installs SIGTERM and SIGINT handling.
type Option struct {
	Start   func(context.Context, Spec) (Running, error)
	Signals <-chan os.Signal
}

type persisted struct {
	Version      int             `json:"version"`
	PrivateKey   string          `json:"privateKey"`
	PresharedKey string          `json:"presharedKey"`
	Region       json.RawMessage `json:"region"`
}

type request struct {
	Version        int             `json:"version"`
	State          json.RawMessage `json:"state"`
	AllowedClients []string        `json:"allowedClients"`
	LocalPort      int             `json:"localPort"`
}

// Run reads one configuration, writes one reply, and stays alive until stdin
// closes or a termination signal arrives. Secrets are written only to out.
func Run(in io.Reader, out io.Writer, opt Option) error {
	reader := bufio.NewReaderSize(in, 64*1024)
	line, err := reader.ReadSlice('\n')
	if err != nil {
		return writeFixed(out, errRequest)
	}
	var req request
	if err := json.Unmarshal(line, &req); err != nil {
		return writeFixed(out, errRequest)
	}
	if req.Version != protoVersion {
		return writeFixed(out, errVersion)
	}
	if req.LocalPort < 1024 || req.LocalPort > 65534 {
		return writeFixed(out, errPort)
	}
	allowed, code := parseAllowlist(req.AllowedClients)
	if code != "" {
		return writeFixed(out, code)
	}
	private, psk, region, code := parseState(req.State)
	if code != "" {
		return writeFixed(out, code)
	}
	if private.IsZero() {
		private = key.NewNode()
		psk = tailcat.NewPresharedKey()
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	watchStop(ctx, cancel, reader, opt.Signals)

	conns, connCtx := newLiveConns(ctx)
	defer conns.close()
	start := opt.Start
	if start == nil {
		start = realStart
	}
	running, err := start(connCtx, Spec{
		Key: private, PSK: psk, Region: region, Allowed: allowed,
		OnTCP: serveVirtual(req.LocalPort, conns),
	})
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return writeFixed(out, errStart)
	}
	defer running.Close()
	if ctx.Err() != nil {
		return ctx.Err()
	}
	chosen := running.Region()
	addr := running.Address()
	if chosen == nil || chosen.RegionID == 0 || len(addr) < 2 || addr[:2] != "tc" {
		return writeFixed(out, errStart)
	}
	state, err := marshalState(private, psk, chosen)
	if err != nil {
		return writeFixed(out, errStart)
	}
	body, err := json.Marshal(struct {
		Version int    `json:"version"`
		Address string `json:"address"`
		Port    int    `json:"port"`
		State   string `json:"state"`
	}{Version: protoVersion, Address: addr, Port: virtualPort, State: state})
	if err != nil {
		return writeFixed(out, errStart)
	}
	if _, err := out.Write(append(body, '\n')); err != nil {
		return err
	}
	<-ctx.Done()
	return nil
}

func watchStop(ctx context.Context, cancel context.CancelFunc, rest io.Reader, signals <-chan os.Signal) {
	if signals == nil {
		ch := make(chan os.Signal, 1)
		signal.Notify(ch, syscall.SIGTERM, syscall.SIGINT)
		go func() {
			<-ctx.Done()
			signal.Stop(ch)
		}()
		signals = ch
	}
	go func() {
		_, _ = io.Copy(io.Discard, rest)
		cancel()
	}()
	go func() {
		select {
		case <-signals:
			cancel()
		case <-ctx.Done():
		}
	}()
}

func parseAllowlist(texts []string) ([]key.NodePublic, string) {
	if len(texts) == 0 {
		return nil, errAllowlist
	}
	out := make([]key.NodePublic, 0, len(texts))
	seen := make(map[string]struct{}, len(texts))
	for _, text := range texts {
		var pub key.NodePublic
		if err := pub.UnmarshalText([]byte(text)); err != nil || pub.IsZero() {
			return nil, errClientKey
		}
		if _, exists := seen[pub.String()]; exists {
			return nil, errAllowlist
		}
		seen[pub.String()] = struct{}{}
		out = append(out, pub)
	}
	return out, ""
}

func parseState(raw json.RawMessage) (key.NodePrivate, tailcat.PresharedKey, *tailcfg.DERPRegion, string) {
	if len(raw) == 0 || string(raw) == "null" {
		return key.NodePrivate{}, tailcat.PresharedKey{}, nil, ""
	}
	var text string
	if err := json.Unmarshal(raw, &text); err != nil || text == "" {
		return key.NodePrivate{}, tailcat.PresharedKey{}, nil, errState
	}
	var doc persisted
	if json.Unmarshal([]byte(text), &doc) != nil || doc.Version != stateVersion {
		return key.NodePrivate{}, tailcat.PresharedKey{}, nil, errState
	}
	var private key.NodePrivate
	if private.UnmarshalText([]byte(doc.PrivateKey)) != nil || private.IsZero() {
		return key.NodePrivate{}, tailcat.PresharedKey{}, nil, errState
	}
	var psk tailcat.PresharedKey
	if psk.UnmarshalText([]byte(doc.PresharedKey)) != nil || psk.IsZero() {
		return key.NodePrivate{}, tailcat.PresharedKey{}, nil, errState
	}
	var region tailcfg.DERPRegion
	if len(doc.Region) == 0 || string(doc.Region) == "null" || json.Unmarshal(doc.Region, &region) != nil || region.RegionID == 0 {
		return key.NodePrivate{}, tailcat.PresharedKey{}, nil, errState
	}
	return private, psk, &region, ""
}

func marshalState(private key.NodePrivate, psk tailcat.PresharedKey, region *tailcfg.DERPRegion) (string, error) {
	privText, err := private.MarshalText()
	if err != nil {
		return "", err
	}
	pskText, err := psk.MarshalText()
	if err != nil {
		return "", err
	}
	regionText, err := json.Marshal(region)
	if err != nil {
		return "", err
	}
	body, err := json.Marshal(persisted{
		Version: stateVersion, PrivateKey: string(privText), PresharedKey: string(pskText), Region: regionText,
	})
	if err != nil {
		return "", err
	}
	return string(body), nil
}

func writeFixed(w io.Writer, code string) error {
	switch code {
	case errRequest, errVersion, errAllowlist, errClientKey, errPort, errState, errStart:
	default:
		code = errStart
	}
	_, err := io.WriteString(w, `{"error":"`+code+"\"}\n")
	if err != nil {
		return err
	}
	return errors.New(code)
}

func serveVirtual(localPort int, conns *liveConns) func(uint16) func(net.Conn) {
	return func(port uint16) func(net.Conn) {
		if port != virtualPort {
			return nil
		}
		return func(client net.Conn) { proxyLoopback(client, localPort, conns) }
	}
}

func proxyLoopback(client net.Conn, localPort int, conns *liveConns) {
	defer client.Close()
	remote, err := conns.dial(localPort)
	if err != nil {
		return
	}
	defer conns.done(remote)
	var wg sync.WaitGroup
	wg.Add(2)
	copy := func(dst, src net.Conn) {
		defer wg.Done()
		_, _ = io.Copy(dst, src)
		_ = dst.Close()
	}
	go copy(remote, client)
	go copy(client, remote)
	wg.Wait()
}

type liveConns struct {
	ctx    context.Context
	cancel context.CancelFunc
	mu     sync.Mutex
	items  map[net.Conn]struct{}
	once   sync.Once
}

func newLiveConns(parent context.Context) (*liveConns, context.Context) {
	ctx, cancel := context.WithCancel(parent)
	return &liveConns{ctx: ctx, cancel: cancel, items: map[net.Conn]struct{}{}}, ctx
}

func (l *liveConns) dial(port int) (net.Conn, error) {
	conn, err := (&net.Dialer{}).DialContext(l.ctx, "tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
	if err != nil {
		return nil, err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.ctx.Err() != nil {
		_ = conn.Close()
		return nil, l.ctx.Err()
	}
	l.items[conn] = struct{}{}
	return conn, nil
}

func (l *liveConns) done(conn net.Conn) {
	_ = conn.Close()
	l.mu.Lock()
	delete(l.items, conn)
	l.mu.Unlock()
}

func (l *liveConns) close() {
	l.once.Do(func() {
		l.cancel()
		l.mu.Lock()
		defer l.mu.Unlock()
		for conn := range l.items {
			_ = conn.Close()
		}
	})
}

type tailServer struct {
	server *tailcat.Server
	region *tailcfg.DERPRegion
	addr   string
}

func (s *tailServer) Address() string             { return s.addr }
func (s *tailServer) Region() *tailcfg.DERPRegion { return s.region }
func (s *tailServer) Close() error                { return s.server.Close() }

func realStart(ctx context.Context, spec Spec) (Running, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	port := spec.Port
	if port == 0 { port = virtualPort }
	server := &tailcat.Server{
		Key:            spec.Key,
		PresharedKey:   spec.PSK,
		Region:         spec.Region,
		AllowedClients: spec.Allowed,
		Logf:           func(string, ...any) {},
		OnTCP:          spec.OnTCP,
		ServedTCPPorts: []filter.PortRange{{First: port, Last: port}},
	}
	errc := make(chan error, 1)
	go func() { errc <- server.Start() }()
	select {
	case <-ctx.Done():
		go func() {
			if err := <-errc; err == nil {
				_ = server.Close()
			}
		}()
		return nil, ctx.Err()
	case err := <-errc:
		if err != nil {
			return nil, err
		}
		if ctx.Err() != nil {
			_ = server.Close()
			return nil, ctx.Err()
		}
		info, err := tailcat.ParseAddr(server.TailcatAddr())
		if err != nil || len(info.Region) != 1 || info.Region[0] == nil || info.Region[0].RegionID == 0 {
			_ = server.Close()
			return nil, errors.New(errStart)
		}
		return &tailServer{server: server, region: info.Region[0], addr: string(server.TailcatAddr())}, nil
	}
}
