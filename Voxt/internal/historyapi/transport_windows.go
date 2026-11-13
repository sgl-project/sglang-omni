//go:build windows

package historyapi

import (
	"bufio"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/netip"
	"os"
	"path/filepath"
	"strings"
)

const endpointVersion = 1

type endpoint struct {
	Version int    `json:"version"`
	Address string `json:"address"`
	Token   string `json:"token"`
}

// Listen makes a loopback listener and puts its address and a random access
// token in path. The data folder's access controls protect the token just as
// they protect the history database.
func Listen(path string) (net.Listener, error) {
	if _, err := os.Stat(path); err == nil {
		if _, err := readEndpoint(path); err != nil {
			return nil, fmt.Errorf("historyapi: not replacing %s: %w", path, err)
		}
	} else if !os.IsNotExist(err) {
		return nil, err
	}
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	tokenBytes := make([]byte, 32)
	if _, err := rand.Read(tokenBytes); err != nil {
		listener.Close()
		return nil, err
	}
	token := hex.EncodeToString(tokenBytes)
	endpointBytes, err := json.Marshal(endpoint{Version: endpointVersion, Address: listener.Addr().String(), Token: token})
	if err != nil {
		listener.Close()
		return nil, err
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".history-endpoint-")
	if err != nil {
		listener.Close()
		return nil, err
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if err := temporary.Chmod(0o600); err == nil {
		_, err = temporary.Write(endpointBytes)
	}
	if closeErr := temporary.Close(); err == nil {
		err = closeErr
	}
	if err == nil {
		err = os.Remove(path)
		if os.IsNotExist(err) {
			err = nil
		}
	}
	if err == nil {
		err = os.Rename(temporaryPath, path)
	}
	if err != nil {
		listener.Close()
		return nil, err
	}
	return &endpointListener{Listener: listener, path: path, token: token, unlinkOnClose: true}, nil
}

type endpointListener struct {
	net.Listener
	path          string
	token         string
	unlinkOnClose bool
}

func (listener *endpointListener) SetUnlinkOnClose(unlink bool) {
	listener.unlinkOnClose = unlink
}

func (listener *endpointListener) Accept() (net.Conn, error) {
	connection, err := listener.Listener.Accept()
	if err != nil {
		return nil, err
	}
	return &authenticatedConn{Conn: connection, reader: bufio.NewReader(connection), token: listener.token}, nil
}

func (listener *endpointListener) Close() error {
	closeErr := listener.Listener.Close()
	if !listener.unlinkOnClose {
		return closeErr
	}
	current, err := readEndpoint(listener.path)
	if err == nil && current.Token == listener.token {
		err = os.Remove(listener.path)
		if os.IsNotExist(err) {
			err = nil
		}
	} else if os.IsNotExist(err) {
		err = nil
	}
	return errors.Join(closeErr, err)
}

type authenticatedConn struct {
	net.Conn
	reader        *bufio.Reader
	token         string
	authenticated bool
}

func (connection *authenticatedConn) Read(buffer []byte) (int, error) {
	if !connection.authenticated {
		line, err := connection.reader.ReadString('\n')
		if err != nil {
			return 0, err
		}
		if subtle.ConstantTimeCompare([]byte(strings.TrimSuffix(line, "\n")), []byte(connection.token)) != 1 {
			return 0, io.EOF
		}
		connection.authenticated = true
	}
	return connection.reader.Read(buffer)
}

func dial(path string) (net.Conn, error) {
	current, err := readEndpoint(path)
	if err != nil {
		return nil, err
	}
	connection, err := net.Dial("tcp4", current.Address)
	if err != nil {
		return nil, err
	}
	if _, err := connection.Write([]byte(current.Token + "\n")); err != nil {
		connection.Close()
		return nil, err
	}
	return connection, nil
}

func readEndpoint(path string) (endpoint, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return endpoint{}, err
	}
	var current endpoint
	if err := json.Unmarshal(data, &current); err != nil {
		return endpoint{}, err
	}
	address, err := netip.ParseAddrPort(current.Address)
	if current.Version != endpointVersion || err != nil || !address.Addr().IsLoopback() {
		return endpoint{}, errors.New("invalid history endpoint")
	}
	token, err := hex.DecodeString(current.Token)
	if err != nil || len(token) != 32 {
		return endpoint{}, errors.New("invalid history endpoint token")
	}
	return current, nil
}
