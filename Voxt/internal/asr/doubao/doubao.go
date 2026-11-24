// Package doubao is Volcengine's Doubao bigmodel streaming speech
// recognition (SAUC, docs/6561/1354869) as an asr.Streamer: a WebSocket
// with a binary framing (codec.go), JSON parameters and results
// (payload.go), and one stream per session (stream.go). Its flash file
// recognition (docs/6561/1631584) is an asr.FileTranscriber (flash.go).
//
// Credentials: the old console's App ID (X-Api-App-Key) and Access Token
// (X-Api-Access-Key), or the new console's single API key (X-Api-Key,
// used when there is no App ID). Either way the model is the resource id
// (X-Api-Resource-Id), which also says how the account is billed
// (".duration": by the hour, ".concurrent": by concurrent streams).
package doubao

import (
	"context"
	"crypto/rand"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// ID is the provider's preset id.
const ID = "doubao"

// Endpoints. DefaultEndpoint is the optimised duplex stream
// (bigmodel_async): partial text while the user speaks, sent only when it
// changes, and with enable_nonstream a second pass that re-recognises each
// finished sentence with the more accurate non-streaming model. The plain
// duplex stream (bigmodel) answers every packet and has no second pass;
// the nostream endpoint answers only every 15 s or at the end, too late
// for live text. Voxt makes the same choice.
const (
	DefaultEndpoint = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
	// DuplexEndpoint is the default of ModelV1, as in Voxt.
	DuplexEndpoint   = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
	NostreamEndpoint = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream"
)

// Models (resource ids).
const (
	// ModelV2 is Doubao streaming ASR 2.0, billed by the hour.
	ModelV2 = "volc.seedasr.sauc.duration"
	// ModelV1 is Doubao streaming ASR 1.0, billed by the hour.
	ModelV1 = "volc.bigasr.sauc.duration"
	// DefaultModel is ModelV2.
	DefaultModel = ModelV2
)

// Models are the built-in choices, as in Voxt, and the flash file
// recognition (ModelFlash), which Voxt 1.x ran but did not list. The
// concurrent-billing resources (volc.seedasr.sauc.concurrent,
// volc.bigasr.sauc.concurrent) work as custom models.
var Models = []string{ModelV2, ModelV1, ModelFlash}

// packet is the audio per packet: the docs advise 100–200 ms and call
// 200 ms the best for the duplex streams.
const packet = 200 * time.Millisecond

// Client opens Doubao streams.
type Client struct {
	endpoint string
	mode     mode
	model    string
	appID    string
	token    string
	http     *http.Client
	// ackTimeout bounds the wait for the first answer (asr.ConnectTimeout),
	// finalTimeout the wait for the final result after Finish
	// (asr.FinalTimeout); tests shorten them.
	ackTimeout, finalTimeout time.Duration
}

var _ asr.Streamer = (*Client)(nil)

// New checks cfg: cfg.Key is required, the Access Token with cfg.AppID
// (old console) or the API key without one (new console); the endpoint,
// if set, must be wss:// (ws:// only on loopback).
func New(cfg asr.Config) (*Client, error) {
	c := &Client{
		model:        strings.TrimSpace(cfg.Model),
		appID:        strings.TrimSpace(cfg.AppID),
		token:        strings.TrimSpace(cfg.Key),
		http:         asr.WSClient(cfg.HTTPClient),
		ackTimeout:   asr.ConnectTimeout,
		finalTimeout: asr.FinalTimeout,
	}
	if err := checkKey(c.appID, c.token); err != nil {
		return nil, err
	}
	if c.model == "" {
		c.model = DefaultModel
	}
	ep := strings.TrimSpace(cfg.Endpoint)
	if ep == "" {
		ep = DefaultEndpoint
		if c.model == ModelV1 {
			ep = DuplexEndpoint
		}
	}
	u, err := asr.CheckEndpoint(ID, ep, "wss")
	if err != nil {
		return nil, err
	}
	c.endpoint, c.mode = u.String(), modeOf(u)
	return c, nil
}

// Caps: streaming, hot words (corpus.context hotwords) and a prompt (the
// dialog context of corpus.context). No documented length limit.
func (c *Client) Caps() asr.Caps {
	return asr.Caps{Stream: true, Hints: true, Prompt: true}
}

// Endpoint is the URL streams connect to.
func (c *Client) Endpoint() string { return c.endpoint }

// Model is the resource id.
func (c *Client) Model() string { return c.model }

// Open connects, sends the parameters and waits for the service to
// accept them (its first answer), so that bad credentials, an unopened
// resource or a refused parameter fail here.
func (c *Client) Open(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	s, err := c.connect(ctx, req)
	if err != nil {
		return nil, asr.HideKey(err, c.token)
	}
	return s, nil
}

func (c *Client) connect(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	if req.SampleRate != 0 && req.SampleRate != asr.SampleRate {
		return nil, asr.Errorf(ID, asr.KindAudio, "takes %d Hz audio only, not %d Hz", asr.SampleRate, req.SampleRate)
	}
	params, err := buildRequest(c.mode, req, strings.TrimSpace(req.Prompt))
	if err != nil {
		return nil, asr.Errorf(ID, asr.KindConfig, "cannot build the request: %v", err)
	}
	id := newUUID()
	h := authHeader(c.appID, c.token)
	h.Set("X-Api-Resource-Id", c.model)
	h.Set("X-Api-Request-Id", id)
	h.Set("X-Api-Connect-Id", id)
	conn, resp, err := asr.Dial(ctx, ID, c.endpoint, asr.WSOptions{Client: c.http, Header: h, ReadLimit: maxFrame})
	if err != nil {
		return nil, handshakeError(err)
	}
	logID := ""
	if resp != nil {
		logID = resp.Header.Get("X-Tt-Logid")
	}
	s := newStream(ctx, c, conn, logID)
	if err := s.start(params); err != nil {
		s.abort()
		return nil, err
	}
	go s.read()
	return s, nil
}

// checkKey says what a setup without its key lacks: the Access Token
// with an App ID, the API key without one.
func checkKey(appID, token string) error {
	switch {
	case token != "":
		return nil
	case appID != "":
		return asr.Errorf(ID, asr.KindConfig, "no Access Token")
	default:
		return asr.Errorf(ID, asr.KindConfig, "no API key")
	}
}

// authHeader is a request's credentials (docs/6561/1354869, 鉴权): the
// old console's App ID and Access Token, or, without an App ID, the new
// console's API key. Both also send the resource and request ids.
func authHeader(appID, token string) http.Header {
	h := http.Header{}
	if appID == "" {
		h.Set("X-Api-Key", token)
		return h
	}
	h.Set("X-Api-App-Key", appID)
	h.Set("X-Api-Access-Key", token)
	return h
}

// newUUID is a random (version 4) UUID.
func newUUID() string {
	var b [16]byte
	rand.Read(b[:])
	b[6] = b[6]&0x0F | 0x40
	b[8] = b[8]&0x3F | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// String is for logs: no credentials.
func (c *Client) String() string {
	u, _ := url.Parse(c.endpoint)
	host := ""
	if u != nil {
		host = u.Host + u.Path
	}
	return fmt.Sprintf("doubao(%s, %s)", host, c.model)
}
