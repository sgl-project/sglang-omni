The `.jsonl` files are what the fake service plays in `golden_test.go`
(the directives are described in `script_test.go`). Their server messages
follow the examples of Alibaba Cloud Model Studio's references, read on
2026-10-15:

- inference protocol: "Fun-ASR-Realtime" WebSocket guide, client events and
  server events (help.aliyun.com/zh/model-studio/fun-asr-realtime-websocket-api,
  …/fun-asr-client-events, …/fun-asr-server-events);
- realtime protocol: "Qwen-ASR-Realtime" client and server events
  (…/qwen-asr-realtime-client-events, …/qwen-asr-realtime-server-events);
- error codes: Model Studio's error code list (InvalidApiKey, Arrearage,
  Throttling.RateQuota);

and how Voxt (@baa6f31, `RemoteASRAliyunStreaming.swift`) reads them. The
recognised texts are made up.

The `.golden` files are what the client sent (JSON with sorted keys, ids
as `<task-id>` / `<event-id>`, audio as its size, repeated lines as `×n`)
and the events it delivered. `VOXTNEW_UPDATE_GOLDEN=1 go test .` rewrites
them; review the diff.
