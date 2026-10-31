module github.com/sgl-project/sglang-omni/VoxtNew

go 1.27.1

tool github.com/egoist/mygo/cmd/mygo

require (
	github.com/coder/websocket v1.8.15
	github.com/ebitengine/purego v0.11.1
	github.com/egoist/mygo v0.5.5
	github.com/hehehai/actnowkit v0.9.1
	golang.org/x/image v0.46.0
	golang.org/x/text v0.43.0
)

require github.com/go-text/typesetting v0.3.5 // indirect

replace github.com/egoist/mygo => ./third_party/mygo
