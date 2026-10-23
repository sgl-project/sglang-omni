package omni

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
)

// VAD is Silero VAD on the runtime (vad.Model): one server shared by
// every detector, started on first use, stopped when idle.
type VAD struct {
	rt   *Runtime
	spec Spec
	dir  string
}

var _ vad.Model = (*VAD)(nil)

// NewVAD is Silero VAD (VADSilero) installed in modelDir (config.json and
// model.safetensors), on rt. Nothing starts yet.
func NewVAD(rt *Runtime, modelDir string) (*VAD, error) {
	spec, _ := Lookup(VADSilero)
	if err := checkDir(modelDir); err != nil {
		return nil, err
	}
	return &VAD{rt: rt, spec: spec, dir: modelDir}, nil
}

// Warm starts the server in the background.
func (v *VAD) Warm() { v.rt.Warm(v.spec, v.dir) }

// Speech finds the speech in a recording (/v1/vad/speech_timestamps, with
// the tuned threshold, minimum speech and silence, and padding).
func (v *VAD) Speech(ctx context.Context, pcm []byte, rate int) ([]vad.Segment, error) {
	if rate != 0 && rate != sampleRate {
		return nil, fmt.Errorf("omni: the voice activity model takes 16 kHz, not %d Hz", rate)
	}
	l, err := v.rt.acquire(ctx, v.spec, v.dir)
	if err != nil {
		return nil, err
	}
	defer l.release()
	t := v.spec.VAD
	var b bytes.Buffer
	w := multipart.NewWriter(&b)
	_ = w.WriteField("threshold", strconv.FormatFloat(t.Threshold, 'g', -1, 64))
	_ = w.WriteField("min_speech_duration_ms", itoa(int(t.MinSpeech/time.Millisecond)))
	_ = w.WriteField("min_silence_duration_ms", itoa(int(t.MinSilence/time.Millisecond)))
	_ = w.WriteField("speech_pad_ms", itoa(int(t.SpeechPad/time.Millisecond)))
	fw, _ := w.CreateFormFile("file", "audio.wav")
	_, _ = fw.Write(asr.WAV(pcm, sampleRate))
	_ = w.Close()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.endpoint().base()+"/v1/vad/speech_timestamps", &b)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	resp, err := loopback.Do(req)
	if err != nil {
		return nil, asr.NetError(ctx, Provider, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return nil, asr.NetError(ctx, Provider, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("omni: voice activity detection answered %d", resp.StatusCode)
	}
	var out struct {
		Timestamps *[]struct {
			Start *int `json:"start"`
			End   *int `json:"end"`
		} `json:"timestamps"`
	}
	if json.Unmarshal(raw, &out) != nil || out.Timestamps == nil {
		return nil, fmt.Errorf("omni: an unreadable voice activity answer")
	}
	segs := make([]vad.Segment, 0, len(*out.Timestamps))
	for _, ts := range *out.Timestamps {
		if ts.Start == nil || ts.End == nil || *ts.Start > *ts.End {
			return nil, fmt.Errorf("omni: an unreadable voice activity range")
		}
		segs = append(segs, vad.Segment{Start: *ts.Start, End: *ts.End})
	}
	return segs, nil
}

// Open starts a stream detector (/v1/vad/stream): the server keeps this
// stream's state; each 512-sample chunk is one exchange.
func (v *VAD) Open(ctx context.Context) (vad.ModelStream, error) {
	// The start and the dial are bounded; ctx then bounds the stream.
	dctx, cancel := context.WithTimeout(ctx, asr.ConnectTimeout)
	defer cancel()
	l, err := v.rt.acquire(dctx, v.spec, v.dir)
	if err != nil {
		return nil, err
	}
	ep := l.endpoint()
	conn, _, err := websocket.Dial(dctx, fmt.Sprintf("ws://%s:%d/v1/vad/stream", ep.host, ep.port), &websocket.DialOptions{HTTPClient: loopback})
	if err != nil {
		l.release()
		return nil, asr.NetError(dctx, Provider, err)
	}
	return &vadStream{ctx: ctx, l: l, conn: conn, t: v.spec.VAD}, nil
}

// vadStream is one stream's detector.
type vadStream struct {
	ctx  context.Context
	l    *lease
	conn *websocket.Conn
	t    *VADTuning
	rest []byte // PCM short of a chunk
	once sync.Once
}

func (s *vadStream) Chunk() int         { return s.t.Chunk }
func (s *vadStream) Threshold() float64 { return s.t.Threshold }

// Feed sends each whole chunk as float32 samples and reads its
// probability (the server answers once per message, with the last
// chunk's).
func (s *vadStream) Feed(pcm []byte) ([]float64, error) {
	s.rest = append(s.rest, pcm...)
	size := s.t.Chunk * 2
	var probs []float64
	msg := make([]byte, s.t.Chunk*4)
	for len(s.rest) >= size {
		for i := range s.t.Chunk {
			v := float32(int16(binary.LittleEndian.Uint16(s.rest[i*2:]))) / 32768
			binary.LittleEndian.PutUint32(msg[i*4:], math.Float32bits(v))
		}
		s.rest = s.rest[size:]
		p, err := s.exchange(msg)
		if err != nil {
			return probs, err
		}
		probs = append(probs, p)
	}
	return probs, nil
}

func (s *vadStream) exchange(msg []byte) (float64, error) {
	ctx, cancel := context.WithTimeout(s.ctx, vad.ModelTimeout)
	defer cancel()
	if err := s.conn.Write(ctx, websocket.MessageBinary, msg); err != nil {
		return 0, asr.NetError(ctx, Provider, err)
	}
	_, raw, err := s.conn.Read(ctx)
	if err != nil {
		return 0, asr.NetError(ctx, Provider, err)
	}
	var reply struct {
		Probability *float64 `json:"probability"`
		Error       string   `json:"error"`
	}
	if json.Unmarshal(raw, &reply) != nil || reply.Error != "" || reply.Probability == nil {
		return 0, fmt.Errorf("omni: voice activity stream: %q", asr.Clip(string(raw), 120))
	}
	return *reply.Probability, nil
}

// Close ends the stream and releases the server.
func (s *vadStream) Close() {
	s.once.Do(func() {
		s.conn.CloseNow()
		s.l.release()
	})
}
