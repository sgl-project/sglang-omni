package doubao

import (
	"bytes"
	"compress/gzip"
	"encoding/hex"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// Wire bytes worked out by hand from the docs' header table.
func TestEncodeBytes(t *testing.T) {
	cases := []struct {
		name string
		f    Frame
		want string
	}{
		{"audio seq 2", Frame{Type: AudioOnlyRequest, Flags: FlagSeq, Seq: 2, Payload: []byte("ab")},
			"11210000" + "00000002" + "00000002" + "6162"},
		{"last audio seq -3, empty", Frame{Type: AudioOnlyRequest, Flags: FlagSeq | FlagLast, Seq: -3},
			"11230000" + "fffffffd" + "00000000"},
		{"last without sequence", Frame{Type: AudioOnlyRequest, Flags: FlagLast, Payload: []byte{1}},
			"11220000" + "00000001" + "01"},
		{"full request json", Frame{Type: FullClientRequest, Flags: FlagSeq, Serialization: SerJSON, Seq: 1, Payload: []byte("{}")},
			"11111000" + "00000001" + "00000002" + "7b7d"},
		{"error frame", Frame{Type: ServerError, Serialization: SerJSON, Code: 45000001, Payload: []byte("x")},
			"11f01000" + "02aea541" + "00000001" + "78"},
		{"server ack", Frame{Type: ServerAck, Flags: FlagSeq, Seq: 1, Payload: []byte("ok")},
			"11b10000" + "00000001" + "6f6b"},
	}
	for _, c := range cases {
		got, err := Encode(c.f)
		if err != nil {
			t.Fatal(err)
		}
		if hex.EncodeToString(got) != c.want {
			t.Errorf("%s: got %x, want %s", c.name, got, c.want)
		}
		back, err := Decode(got)
		if err != nil {
			t.Fatalf("%s: decode: %v", c.name, err)
		}
		if back.Type != c.f.Type || back.Flags != c.f.Flags || back.Seq != c.f.Seq || back.Code != c.f.Code || !bytes.Equal(back.Payload, c.f.Payload) {
			t.Errorf("%s: round trip %+v, want %+v", c.name, back, c.f)
		}
	}
}

func TestRoundTripGzipAndEvent(t *testing.T) {
	f := Frame{Type: FullServerResponse, Flags: FlagSeq | FlagLast | FlagEvent, Serialization: SerJSON, Compression: CompGzip, Seq: -5, Event: 150, Payload: []byte(`{"result":{"text":"好"}}`)}
	b, err := Encode(f)
	if err != nil {
		t.Fatal(err)
	}
	g, err := Decode(b)
	if err != nil {
		t.Fatal(err)
	}
	if g.Seq != -5 || g.Event != 150 || !g.Last() || string(g.Payload) != string(f.Payload) {
		t.Fatalf("got %+v", g)
	}
	// An empty gzip payload decodes to nothing.
	b, _ = Encode(Frame{Type: FullServerResponse, Flags: FlagSeq, Compression: CompGzip, Seq: 1})
	if g, err := Decode(b); err != nil || len(g.Payload) != 0 {
		t.Fatalf("empty gzip: %+v %v", g, err)
	}
}

func TestLast(t *testing.T) {
	for _, c := range []struct {
		f    Frame
		want bool
	}{
		{Frame{Flags: FlagSeq, Seq: 3}, false},
		{Frame{Flags: FlagSeq, Seq: -3}, true},
		{Frame{Flags: FlagLast}, true},
		{Frame{Flags: 0, Seq: -1}, false}, // no sequence field: Seq means nothing
	} {
		if c.f.Last() != c.want {
			t.Errorf("%+v: Last %v", c.f, !c.want)
		}
	}
}

func TestDecodeErrors(t *testing.T) {
	h := func(s string) []byte { b, _ := hex.DecodeString(s); return b }
	var bomb bytes.Buffer
	w := gzip.NewWriter(&bomb)
	w.Write(make([]byte, maxPayload+1))
	w.Close()
	bombFrame, _ := Encode(Frame{Type: FullServerResponse, Flags: FlagSeq, Seq: 1})
	bombFrame = append(bombFrame[:8], h("00000000")...)
	bombFrame[2] = SerJSON<<4 | CompGzip
	bombFrame = append(bombFrame[:8], append(beU32(uint32(bomb.Len())), bomb.Bytes()...)...)

	cases := map[string][]byte{
		"short":               h("1190"),
		"version 2":           h("2190000000000001" + "00000000"),
		"header size 0":       h("1090000000000001" + "00000000"),
		"header past end":     h("1f900000"),
		"unknown type":        h("11500000" + "00000000"),
		"no sequence":         h("119100" + "00" + "0000"),
		"no payload size":     h("11910000" + "00000001"),
		"payload past end":    h("11910000" + "00000001" + "00000010" + "7b7d"),
		"no error code":       h("11f00000" + "0000"),
		"unknown compression": h("11910200" + "00000001" + "00000002" + "7b7d"),
		"bad gzip":            h("11911100" + "00000001" + "00000004" + "deadbeef"),
		"truncated gzip":      append(h("11911100"+"00000001"+"0000000a"), bomb.Bytes()[:10]...),
		"gzip bomb":           bombFrame,
		"too large":           make([]byte, maxFrame+1),
	}
	for name, b := range cases {
		_, err := Decode(b)
		if asr.KindOf(err) != asr.KindProtocol {
			t.Errorf("%s: got %v, want a protocol error", name, err)
		}
	}
}

func beU32(v uint32) []byte { return []byte{byte(v >> 24), byte(v >> 16), byte(v >> 8), byte(v)} }

func FuzzDecode(f *testing.F) {
	for _, s := range []string{"11910000000000010000000000", "11f0100002aea5410000000178", "11b100000000000100"} {
		b, _ := hex.DecodeString(s)
		f.Add(b)
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		fr, err := Decode(b)
		if err != nil {
			if asr.KindOf(err) != asr.KindProtocol {
				t.Fatalf("error %v is not a protocol error", err)
			}
			return
		}
		if len(fr.Payload) > maxPayload {
			t.Fatal("payload over the limit")
		}
	})
}
