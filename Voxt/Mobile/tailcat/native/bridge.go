package main

import (
	"context"
	"encoding/json"
	"net"
	"strconv"

	"github.com/tailscale/tailcat"
	"tailscale.com/types/key"
)

func Start(address, clientKey string, remotePort int) string {
	if !validAddress(address) {
		return `{"error":"invalid-address"}`
	}
	private, err := parseClientKey(clientKey)
	if err != nil {
		return `{"error":"invalid-client-key"}`
	}
	if remotePort < 1 || remotePort > 65535 {
		return `{"error":"invalid-port"}`
	}
	opened := newSessionDial(address, private, remotePort)
	session := &bridgeSession{close: opened.close}
	forward, err := openForward(opened.dial)
	if err != nil {
		session.Close()
		return `{"error":"bind-failed"}`
	}
	session.forward = forward
	port, err := loopbackPort(forward)
	if err != nil {
		session.Close()
		return `{"error":"bind-failed"}`
	}
	handle := register(session)
	body, err := json.Marshal(struct {
		Handle int64 `json:"handle"`
		Port   int   `json:"port"`
	}{
		Handle: handle,
		Port:   port,
	})
	if err != nil {
		Stop(handle)
		return `{"error":"start-failed"}`
	}
	return string(body)
}

func validAddress(address string) bool {
	if address == "" {
		return false
	}
	info, err := tailcat.ParseAddr(tailcat.Addr(address))
	if err != nil {
		return false
	}
	return !info.ServerPublic.IsZero() && !info.ServerDiscoPublic.IsZero()
}

func parseClientKey(text string) (key.NodePrivate, error) {
	var private key.NodePrivate
	if err := private.UnmarshalText([]byte(text)); err != nil {
		return key.NodePrivate{}, err
	}
	if private.IsZero() {
		return key.NodePrivate{}, errZeroKey
	}
	return private, nil
}

func loopbackPort(forward *Forward) (int, error) {
	host, portText, err := net.SplitHostPort(forward.Addr())
	if err != nil || host != "127.0.0.1" {
		return 0, errZeroKey
	}
	port, err := strconv.Atoi(portText)
	if err != nil || port < 1 || port > 65535 {
		return 0, errZeroKey
	}
	return port, nil
}

type sessionDial struct {
	dial  func(context.Context) (net.Conn, error)
	close func()
}

func realSessionDial(address string, clientKey key.NodePrivate, remotePort int) sessionDial {
	client := &tailcat.Client{
		Server: tailcat.Addr(address),
		Key:    clientKey,
		Logf:   func(string, ...any) {},
	}
	port := uint16(remotePort)
	return sessionDial{
		dial: func(ctx context.Context) (net.Conn, error) {
			return client.DialTCPPort(ctx, port)
		},
		close: func() { _ = client.Close() },
	}
}

var newSessionDial = realSessionDial
var openForward = startForward
