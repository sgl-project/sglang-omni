package omnitest

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
)

// speechRMS is the level over which the fake hears speech (-34 dBFS).
const speechRMS = 0.02

// Main runs the fake server and exits.
func Main() {
	o := opts()
	args := os.Args[1:]
	// The flag checks come first and record nothing, as the real server's
	// argument parsing fails before it serves (Runtime's probe).
	if limit := arg(args, "--cache-limit-mb"); limit != "" {
		if mb, err := strconv.Atoi(limit); o.Older || err != nil || mb < -1 {
			if o.Older {
				fmt.Fprintln(os.Stderr, "unknown argument --cache-limit-mb")
			} else {
				fmt.Fprintln(os.Stderr, "--cache-limit-mb must be 0 to 1073741824, or -1 for MLX's limit")
			}
			os.Exit(2)
		}
	}
	record(o, Record{Event: "launch", Args: args})
	emit := func(v map[string]any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(os.Stdout, "%s\n", b)
	}
	switch o.Mode {
	case ModeFail:
		emit(map[string]any{"event": "failed", "reason": "model load failed: fake"})
		os.Exit(1)
	case ModeHang:
		for {
			time.Sleep(time.Hour) // (a bare select{} is a deadlock to the runtime)
		}
	}
	if arg(args, "--model-directory") == "" {
		fmt.Fprintln(os.Stderr, "--model-path is required")
		os.Exit(2)
	}
	time.Sleep(o.Load)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		emit(map[string]any{"event": "failed", "reason": err.Error()})
		os.Exit(1)
	}
	kind := arg(args, "--model-kind")
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"status": "healthy", "running": true})
	})
	if kind == "silero_vad" {
		mux.HandleFunc("/v1/vad/speech_timestamps", func(w http.ResponseWriter, r *http.Request) { timestamps(o, w, r) })
		mux.HandleFunc("/v1/vad/stream", func(w http.ResponseWriter, r *http.Request) { vadStream(o, w, r) })
	} else {
		mux.HandleFunc("/v1/audio/transcriptions", func(w http.ResponseWriter, r *http.Request) { final(o, w, r) })
		mux.HandleFunc("/v1/realtime", func(w http.ResponseWriter, r *http.Request) { realtime(o, w, r) })
	}
	go func() { _ = http.Serve(ln, mux) }()
	emit(map[string]any{"event": "ready", "host": "127.0.0.1", "port": ln.Addr().(*net.TCPAddr).Port,
		"model_name": "fake-" + kind, "server_pid": os.Getpid(), "startup_s": 0.01})
	sc := bufio.NewScanner(os.Stdin)
	for sc.Scan() {
		if strings.Contains(sc.Text(), `"shutdown"`) {
			record(o, Record{Event: "shutdown"})
			emit(map[string]any{"event": "stopped"})
			os.Exit(0)
		}
	}
	os.Exit(0) // stdin closed: the owner is gone
}

// finals counts the Finals this server answered.
var finals atomic.Int32

// final answers a transcription: heartbeats while it "decodes", then the
// done event with the generation metadata.
func final(o Opts, w http.ResponseWriter, r *http.Request) {
	n := int(finals.Add(1))
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		http.Error(w, `{"detail":"bad form"}`, http.StatusBadRequest)
		return
	}
	fields := map[string]string{}
	for k, v := range r.MultipartForm.Value {
		fields[k] = v[0]
	}
	samples := 0
	if f, _, err := r.FormFile("file"); err == nil {
		b, _ := io.ReadAll(f)
		samples = max(len(b)-44, 0) / 2
	}
	record(o, Record{Event: "final", Fields: fields, Samples: samples})
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	fl, _ := w.(http.Flusher)
	flush := func() {
		if fl != nil {
			fl.Flush()
		}
	}
	flush()
	switch o.Mode {
	case ModeStall:
		<-r.Context().Done()
		return
	case ModeCrashFinal:
		os.Exit(3)
	}
	for end := time.Now().Add(o.Delay); time.Now().Before(end); time.Sleep(20 * time.Millisecond) {
		_, _ = io.WriteString(w, ":\n\n")
		flush()
	}
	if o.Mode == ModeError || o.FailFrom > 0 && n >= o.FailFrom {
		_, _ = io.WriteString(w, `data: {"type":"error","error":{"type":"server_error","code":"transcription_failed","message":"Transcription failed."}}`+"\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		return
	}
	language := fields["language"]
	if language == "" {
		language = "Chinese"
	}
	text := strings.ReplaceAll(o.Text, "{n}", strconv.Itoa(n))
	if o.Echo && fields["prompt"] != "" {
		text += strings.Join(strings.Split(fields["prompt"], "\n"), "，") + "。"
	}
	done, _ := json.Marshal(map[string]any{"type": "transcript.text.done", "text": text,
		"generation_metadata": map[string]any{"generated_token_count": 10, "language": language, "finish_reason": "stop"}})
	fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", done)
	flush()
}

// realtime is a manual-turn session: a partial after the first 100 ms of
// audio and then each second, the final segment on commit.
func realtime(o Opts, w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer c.CloseNow()
	ctx := r.Context()
	idx := 0
	send := func(v map[string]any) {
		idx++
		v["event_index"] = idx
		b, _ := json.Marshal(v)
		_ = c.Write(ctx, websocket.MessageText, b)
	}
	samples, next, language := 0, 1600, ""
	defer func() {
		record(o, Record{Event: "live", Samples: samples, Fields: map[string]string{"language": language}})
	}()
	for {
		_, b, err := c.Read(ctx)
		if err != nil {
			return
		}
		var ev struct {
			Type    string         `json:"type"`
			Audio   string         `json:"audio"`
			Session map[string]any `json:"session"`
		}
		_ = json.Unmarshal(b, &ev)
		switch ev.Type {
		case "session.update":
			language, _ = ev.Session["language"].(string)
			send(map[string]any{"type": "transcription_session.updated"})
		case "input_audio_buffer.append":
			pcm, err := base64.StdEncoding.DecodeString(ev.Audio)
			if err != nil {
				send(map[string]any{"type": "error", "error": map[string]any{"code": "invalid_audio"}})
				continue
			}
			samples += len(pcm) / 2
			if samples >= next {
				next = samples + 16000
				send(map[string]any{"type": "transcription.segment", "segment_id": 0, "text": fmt.Sprintf("partial %d", samples/1600), "is_final": false})
			}
		case "input_audio_buffer.commit":
			send(map[string]any{"type": "transcription.segment", "segment_id": 0, "text": "live final", "is_final": true})
		case "transcription.done":
			send(map[string]any{"type": "transcription.completed", "text": "live final"})
			return
		}
	}
}

// pcmRMS is the level of PCM16 (0–1).
func pcmRMS(pcm []byte) float64 {
	n := len(pcm) / 2
	if n == 0 {
		return 0
	}
	var sum float64
	for i := range n {
		v := float64(int16(binary.LittleEndian.Uint16(pcm[i*2:]))) / 32768
		sum += v * v
	}
	return math.Sqrt(sum / float64(n))
}

// timestamps finds the 512-sample chunks over speechRMS, joined into
// ranges.
func timestamps(o Opts, w http.ResponseWriter, r *http.Request) {
	if o.Mode == ModeVADError {
		http.Error(w, `{"detail":"Voice activity detection failed."}`, http.StatusInternalServerError)
		return
	}
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		http.Error(w, `{"detail":"bad form"}`, http.StatusBadRequest)
		return
	}
	f, _, err := r.FormFile("file")
	if err != nil {
		http.Error(w, `{"detail":"file is required"}`, http.StatusBadRequest)
		return
	}
	b, _ := io.ReadAll(f)
	pcm := b[min(44, len(b)):]
	fields := map[string]string{}
	for k, v := range r.MultipartForm.Value {
		fields[k] = v[0]
	}
	record(o, Record{Event: "vad-timestamps", Fields: fields, Samples: len(pcm) / 2})
	type ts struct {
		Start int `json:"start"`
		End   int `json:"end"`
	}
	out := []ts{}
	for i := 0; (i+1)*1024 <= len(pcm); i++ {
		if pcmRMS(pcm[i*1024:(i+1)*1024]) < speechRMS {
			continue
		}
		if n := len(out); n > 0 && out[n-1].End == i*512 {
			out[n-1].End = (i + 1) * 512
		} else {
			out = append(out, ts{i * 512, (i + 1) * 512})
		}
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"sample_rate": 16000, "timestamps": out})
}

// vadStream answers each binary message of float32 samples with a
// probability: 0.9 when its last 512 samples are loud, else 0.05.
func vadStream(o Opts, w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer c.CloseNow()
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	total := 0
	defer func() { record(o, Record{Event: "vad-stream", Samples: total}) }()
	for {
		_, b, err := c.Read(ctx)
		if err != nil {
			return
		}
		if o.Mode == ModeVADError && total > 8000 {
			return // the stream breaks after half a second
		}
		n := len(b) / 4
		total += n
		var sum float64
		k := min(n, 512)
		for i := n - k; i < n; i++ {
			v := float64(math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:])))
			sum += v * v
		}
		p := 0.05
		if k > 0 && math.Sqrt(sum/float64(k)) >= speechRMS {
			p = 0.9
		}
		_ = c.Write(ctx, websocket.MessageText, []byte(fmt.Sprintf(`{"probability":%g}`, p)))
	}
}
