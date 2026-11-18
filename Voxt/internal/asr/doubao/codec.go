package doubao

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// The binary framing of the SAUC (bigmodel streaming ASR) WebSocket API,
// as Volcengine documents it (docs/6561/1354869, "WebSocket 二进制协议"):
//
//	byte 0: protocol version (4 bits, 1) | header size in 4-byte words (4 bits, 1)
//	byte 1: message type (4 bits)        | message type specific flags (4 bits)
//	byte 2: serialization (4 bits)       | compression (4 bits)
//	byte 3: reserved
//	[header extensions: (header size - 1) × 4 bytes]
//	[sequence, int32 big-endian]          when flags has FlagSeq
//	[event, int32 big-endian]             when flags has FlagEvent
//	[error code, uint32 big-endian]       error frames only
//	payload size, uint32 big-endian (absent in server ACKs)
//	payload (compressed as byte 2 says)
//
// Every integer is big-endian. Client frames are the full client request
// (JSON parameters) and audio-only requests (raw PCM); the server answers
// with full server responses (JSON results) or error frames.

// MsgType is a frame's message type.
type MsgType uint8

const (
	FullClientRequest  MsgType = 0x1
	AudioOnlyRequest   MsgType = 0x2
	FullServerResponse MsgType = 0x9
	ServerAck          MsgType = 0xB
	ServerError        MsgType = 0xF
)

func (t MsgType) String() string {
	switch t {
	case FullClientRequest:
		return "full-client-request"
	case AudioOnlyRequest:
		return "audio-only-request"
	case FullServerResponse:
		return "full-server-response"
	case ServerAck:
		return "server-ack"
	case ServerError:
		return "server-error"
	}
	return fmt.Sprintf("type-%#x", uint8(t))
}

// Message type specific flags. FlagSeq|FlagLast (0b0011) is the last
// packet with a (negative) sequence; FlagLast alone is the last packet
// without one.
const (
	FlagSeq   uint8 = 0x1
	FlagLast  uint8 = 0x2
	FlagEvent uint8 = 0x4
)

// Serialization and compression methods.
const (
	SerNone  uint8 = 0x0
	SerJSON  uint8 = 0x1
	CompNone uint8 = 0x0
	CompGzip uint8 = 0x1
)

const (
	protocolVersion = 0x1
	// maxFrame bounds one frame from the server (Dial's read limit is the
	// same); maxPayload what a payload may inflate to.
	maxFrame   = 4 << 20
	maxPayload = 8 << 20
)

// Frame is one decoded (or to be encoded) message.
type Frame struct {
	Type          MsgType
	Flags         uint8
	Serialization uint8
	Compression   uint8
	Seq           int32  // when Flags&FlagSeq
	Event         int32  // when Flags&FlagEvent
	Code          uint32 // ServerError frames
	// Payload is uncompressed: Encode compresses it as Compression says,
	// Decode inflates it.
	Payload []byte
}

// Last reports whether the frame is the last of its direction: the last
// flag, or a negative sequence.
func (f Frame) Last() bool {
	return f.Flags&FlagLast != 0 || (f.Flags&FlagSeq != 0 && f.Seq < 0)
}

// Encode builds the wire bytes of f, compressing its payload.
func Encode(f Frame) ([]byte, error) {
	payload := f.Payload
	switch f.Compression {
	case CompNone:
	case CompGzip:
		var b bytes.Buffer
		w := gzip.NewWriter(&b)
		if _, err := w.Write(payload); err != nil {
			return nil, err
		}
		if err := w.Close(); err != nil {
			return nil, err
		}
		payload = b.Bytes()
	default:
		return nil, fmt.Errorf("doubao: unknown compression %d", f.Compression)
	}
	out := make([]byte, 0, 16+len(payload))
	out = append(out,
		protocolVersion<<4|1,
		uint8(f.Type)<<4|f.Flags&0x0F,
		f.Serialization<<4|f.Compression&0x0F,
		0)
	if f.Flags&FlagSeq != 0 {
		out = binary.BigEndian.AppendUint32(out, uint32(f.Seq))
	}
	if f.Flags&FlagEvent != 0 {
		out = binary.BigEndian.AppendUint32(out, uint32(f.Event))
	}
	if f.Type == ServerError {
		out = binary.BigEndian.AppendUint32(out, f.Code)
	}
	if f.Type != ServerAck {
		out = binary.BigEndian.AppendUint32(out, uint32(len(payload)))
	}
	return append(out, payload...), nil
}

// Decode parses one frame and inflates its payload. Anything malformed
// (truncated, an unknown version or compression, bad gzip, too large) is a
// KindProtocol *asr.Error.
func Decode(b []byte) (Frame, error) {
	var f Frame
	if len(b) > maxFrame {
		return f, protoErr("frame of %d bytes is too large", len(b))
	}
	if len(b) < 4 {
		return f, protoErr("truncated frame: %d bytes", len(b))
	}
	if v := b[0] >> 4; v != protocolVersion {
		return f, protoErr("unknown protocol version %d", v)
	}
	hsize := int(b[0]&0x0F) * 4
	if hsize < 4 || hsize > len(b) {
		return f, protoErr("bad header size %d", hsize)
	}
	f.Type = MsgType(b[1] >> 4)
	f.Flags = b[1] & 0x0F
	f.Serialization = b[2] >> 4
	f.Compression = b[2] & 0x0F
	switch f.Type {
	case FullClientRequest, AudioOnlyRequest, FullServerResponse, ServerAck, ServerError:
	default:
		return f, protoErr("unknown message type %#x", uint8(f.Type))
	}
	p := b[hsize:]
	u32 := func(what string) (uint32, error) {
		if len(p) < 4 {
			return 0, protoErr("truncated frame: no %s", what)
		}
		v := binary.BigEndian.Uint32(p)
		p = p[4:]
		return v, nil
	}
	if f.Flags&FlagSeq != 0 {
		v, err := u32("sequence")
		if err != nil {
			return f, err
		}
		f.Seq = int32(v)
	}
	if f.Flags&FlagEvent != 0 {
		v, err := u32("event")
		if err != nil {
			return f, err
		}
		f.Event = int32(v)
	}
	if f.Type == ServerError {
		v, err := u32("error code")
		if err != nil {
			return f, err
		}
		f.Code = v
	}
	raw := p
	if f.Type != ServerAck {
		n, err := u32("payload size")
		if err != nil {
			return f, err
		}
		if uint64(n) > uint64(len(p)) {
			return f, protoErr("truncated frame: payload of %d bytes, %d left", n, len(p))
		}
		raw = p[:n]
	}
	switch f.Compression {
	case CompNone:
		f.Payload = raw
	case CompGzip:
		out, err := gunzip(raw)
		if err != nil {
			return f, err
		}
		f.Payload = out
	default:
		return f, protoErr("unknown compression %d", f.Compression)
	}
	return f, nil
}

// gunzip inflates a gzip payload of at most maxPayload bytes. An empty
// payload stays empty (some frames carry none).
func gunzip(raw []byte) ([]byte, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	r, err := gzip.NewReader(bytes.NewReader(raw))
	if err != nil {
		return nil, protoErr("bad gzip payload: %v", err)
	}
	out, err := io.ReadAll(io.LimitReader(r, maxPayload+1))
	if err != nil {
		return nil, protoErr("bad gzip payload: %v", err)
	}
	if len(out) > maxPayload {
		return nil, protoErr("payload inflates beyond %d bytes", maxPayload)
	}
	return out, nil
}

func protoErr(format string, args ...any) *asr.Error {
	return asr.Errorf(ID, asr.KindProtocol, format, args...)
}
