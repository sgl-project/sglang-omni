# @mygo-plugins/websocket

A `WebSocket` for [MyGo](https://github.com/egoist/mygo) apps whose
connections Go makes, with the API of the browser's. Unlike the browser's,
it may send any header with the opening handshake, such as Authorization,
Cookie or Origin, and is not subject to the page's Content Security Policy.

Use the plugin in Go:

```go
import "github.com/egoist/mygo/plugins/websocket"

mygo.Use(websocket.Plugin) // or websocket.New(websocket.Options{...})
```

and connect in the frontend:

```ts
import { WebSocket } from "@mygo-plugins/websocket";

const ws = new WebSocket("wss://example.com/live", ["v1"], {
  headers: { authorization: `Bearer ${token}` },
});
ws.onopen = () => ws.send("hello");
ws.onmessage = (e) => console.log(e.data);
ws.onclose = (e) => console.log(e.code, e.reason, e.wasClean);
```

Messages are sent in order, `bufferedAmount` counts the bytes Go has not
written yet, and received messages wait in Go while the page is behind, so a
fast server cannot fill the page's memory. Connections close (1001, going
away) when their page navigates away or its window closes. No extensions
(such as permessage-deflate) are negotiated.

[The documentation](https://github.com/egoist/mygo/blob/main/docs/plugins/websocket.md)
covers closing, failures and the options of the Go side.
