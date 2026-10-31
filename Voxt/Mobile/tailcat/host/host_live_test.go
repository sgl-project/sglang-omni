package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/tailscale/tailcat"
	"tailscale.com/types/key"
	"tailscale.com/types/logger"
)

func TestLiveHelperRestartAndRevocation(t *testing.T) {
	helper := os.Getenv("VOXT_TAILCAT_HOST_HELPER")
	if helper == "" {
		t.Skip("set VOXT_TAILCAT_HOST_HELPER to run the live helper test")
	}
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "owned-fixture-ok")
	}))
	defer origin.Close()
	_, portText, err := net.SplitHostPort(origin.Listener.Addr().String())
	if err != nil {
		t.Fatal("fixture address could not be parsed")
	}
	localPort, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal("fixture port could not be parsed")
	}
	firstKey := key.NewNode()
	secondKey := key.NewNode()
	first, stopFirst := launchLiveHelper(t, helper, nil, []string{firstKey.Public().String()}, localPort)
	defer stopFirst()
	checkLiveHTTP(t, first.Address, firstKey, true)
	stopFirst()

	second, stopSecond := launchLiveHelper(t, helper, &first.State, []string{firstKey.Public().String()}, localPort)
	defer stopSecond()
	if second.Address != first.Address {
		t.Fatal("saved host state changed the previous invitation address")
	}
	checkLiveHTTP(t, first.Address, firstKey, true)
	checkLiveHTTP(t, first.Address, secondKey, false)
	stopSecond()

	third, stopThird := launchLiveHelper(t, helper, &second.State, []string{secondKey.Public().String()}, localPort)
	defer stopThird()
	if third.Address != first.Address {
		t.Fatal("allowlist restart changed the previous invitation address")
	}
	checkLiveHTTP(t, first.Address, firstKey, false)
	checkLiveHTTP(t, first.Address, secondKey, true)
}

func launchLiveHelper(t *testing.T, executable string, state *string, allowed []string, localPort int) (struct {
	Address string `json:"address"`
	State   string `json:"state"`
}, func()) {
	t.Helper()
	command := exec.Command(executable)
	input, err := command.StdinPipe()
	if err != nil {
		t.Fatal("helper stdin unavailable")
	}
	output, err := command.StdoutPipe()
	if err != nil {
		t.Fatal("helper stdout unavailable")
	}
	command.Stderr = io.Discard
	if err := command.Start(); err != nil {
		t.Fatal("helper could not start")
	}
	var stopOnce sync.Once
	stop := func() {
		stopOnce.Do(func() {
			_ = input.Close()
			finished := make(chan struct{})
			go func() { _ = command.Wait(); close(finished) }()
			select {
			case <-finished:
			case <-time.After(3 * time.Second):
				_ = command.Process.Kill()
				<-finished
			}
		})
	}
	request, _ := json.Marshal(struct {
		Version        int      `json:"version"`
		State          *string  `json:"state"`
		AllowedClients []string `json:"allowedClients"`
		LocalPort      int      `json:"localPort"`
	}{1, state, allowed, localPort})
	if _, err := input.Write(append(request, '\n')); err != nil {
		stop()
		t.Fatal("helper rejected the launch request")
	}
	lines := make(chan []byte, 1)
	go func() { line, _ := bufio.NewReader(output).ReadBytes('\n'); lines <- line }()
	var line []byte
	select {
	case line = <-lines:
	case <-time.After(20 * time.Second):
		stop()
		t.Fatal("helper did not become ready")
	}
	var ready struct {
		Address string `json:"address"`
		State   string `json:"state"`
	}
	if json.Unmarshal(line, &ready) != nil || ready.Address == "" || ready.State == "" {
		stop()
		t.Fatal("helper did not return a valid ready response")
	}
	return ready, stop
}

func checkLiveHTTP(t *testing.T, address string, identity key.NodePrivate, shouldConnect bool) {
	t.Helper()
	client := tailcat.NewClient(tailcat.Addr(address))
	client.Key = identity
	client.Logf = logger.Discard
	defer client.Close()
	port := uint16(virtualPort)
	if os.Getenv("VOXT_TEST_WRONG_PORT") == "1" {
		port++
	}
	context, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	connection, err := client.DialTCPPort(context, port)
	if !shouldConnect {
		if err == nil {
			_ = connection.Close()
			t.Fatal("revoked or unknown identity connected")
		}
		return
	}
	if err != nil {
		t.Fatal("previous invitation did not connect")
	}
	defer connection.Close()
	if _, err := io.WriteString(connection, "GET / HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\n\r\n"); err != nil {
		t.Fatal("request did not reach the helper")
	}
	response, err := http.ReadResponse(bufio.NewReader(connection), nil)
	if err != nil {
		t.Fatal("helper did not return HTTP")
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(response.Body)
	if response.StatusCode != http.StatusOK || string(body) != "owned-fixture-ok" {
		t.Fatal("helper did not forward the fixture response")
	}
}
