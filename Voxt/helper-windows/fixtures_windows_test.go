//go:build windows

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

type windowsFixtureLine struct {
	To      string          `json:"to"`
	Message json.RawMessage `json:"msg"`
}

func TestFixtureParity(t *testing.T) {
	paths, err := filepath.Glob("../helper/Fixtures/*.jsonl")
	if err != nil || len(paths) < 9 {
		t.Fatalf("fixture transcripts: %v, %v", paths, err)
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			file, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer file.Close()
			pending := make(map[string]string)
			scanner := bufio.NewScanner(file)
			scanner.Buffer(make([]byte, 64*1024), proto.MaxLine)
			lineNumber := 0
			for scanner.Scan() {
				lineNumber++
				if len(bytes.TrimSpace(scanner.Bytes())) == 0 {
					continue
				}
				where := fmt.Sprintf("%s:%d", filepath.Base(path), lineNumber)
				var line windowsFixtureLine
				strictFixtureDecode(t, where+" envelope", scanner.Bytes(), &line)
				if line.To != "helper" && line.To != "app" {
					t.Fatalf("%s: invalid direction %q", where, line.To)
				}
				message, err := proto.Decode(line.Message)
				if err != nil {
					t.Fatalf("%s: %v", where, err)
				}
				encoded, err := proto.Encode(message)
				if err != nil || !fixtureJSONEqual(line.Message, bytes.TrimSuffix(encoded, []byte("\n"))) {
					t.Fatalf("%s: wire envelope does not round-trip", where)
				}
				switch {
				case line.To == "helper" && message.Kind() == proto.KindRequest:
					pending[message.ID] = message.Method
					params := proto.ParamsFor(message.Method)
					if params != nil && slices.Contains(proto.Requests, message.Method) {
						strictFixtureDecode(t, where+" params", message.Params, params)
					}
				case line.To == "app" && message.Kind() == proto.KindNotification:
					params := proto.ParamsFor(message.Method)
					if params == nil || !slices.Contains(proto.Notifications, message.Method) {
						t.Fatalf("%s: unknown notification %s", where, message.Method)
					}
					strictFixtureDecode(t, where+" params", message.Params, params)
				case line.To == "app" && message.Kind() == proto.KindResponse:
					method, found := pending[message.ID]
					if !found {
						t.Fatalf("%s: response to unknown request %s", where, message.ID)
					}
					delete(pending, message.ID)
					if message.Error == nil {
						result := proto.ResultFor(method)
						if result == nil {
							t.Fatalf("%s: no result type for %s", where, method)
						}
						strictFixtureDecode(t, where+" result", message.Result, result)
					}
				default:
					t.Fatalf("%s: %s message cannot go to %s", where, message.Kind(), line.To)
				}
			}
			if err := scanner.Err(); err != nil {
				t.Fatal(err)
			}
			if len(pending) != 0 {
				t.Fatalf("unanswered requests: %v", pending)
			}
		})
	}
}

func strictFixtureDecode(t *testing.T, where string, raw json.RawMessage, destination any) {
	t.Helper()
	if len(raw) == 0 {
		raw = json.RawMessage(`{}`)
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		t.Fatalf("%s: %v", where, err)
	}
	encoded, err := json.Marshal(destination)
	if err != nil {
		t.Fatal(err)
	}
	if !fixtureJSONEqual(raw, encoded) {
		t.Fatalf("%s: value does not round-trip\nwire: %s\ngo:   %s", where, raw, encoded)
	}
}

func fixtureJSONEqual(first, second []byte) bool {
	var firstValue, secondValue any
	if json.Unmarshal(first, &firstValue) != nil || json.Unmarshal(second, &secondValue) != nil {
		return false
	}
	return reflect.DeepEqual(firstValue, secondValue)
}
