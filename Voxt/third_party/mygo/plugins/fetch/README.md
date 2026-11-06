# @mygo-plugins/fetch

A `fetch` for [MyGo](https://github.com/egoist/mygo) apps that makes HTTP
requests from Go: not subject to CORS, free to set any header (Origin,
Cookie, User-Agent, ...), with the standard options and a standard
`Response` whose body streams as Go reads it.

Use the plugin in Go:

```go
import "github.com/egoist/mygo/plugins/fetch"

mygo.Use(fetch.Plugin) // or fetch.New(fetch.Options{Client: ..., Allow: ...})
```

and fetch in the frontend:

```ts
import { fetch } from "@mygo-plugins/fetch";

const ctrl = new AbortController();
const res = await fetch("https://api.example.com/items", {
  headers: { authorization: `Bearer ${token}` },
  signal: ctrl.signal, // aborting cancels the request in Go
});
const items = await res.json();
```

- Aborting the signal, canceling the body, navigating away or closing the
  window cancels the request in Go.
- `redirect: "manual"` returns the redirect response itself, with its
  `Location`, instead of an opaque response.
- `mode`, `credentials`, `cache` and the like do not apply; cookies are those
  of the Go side's `http.Client` (give it a `Jar` to keep them).
- Every response header is readable but `Set-Cookie`, which a `Response`
  cannot hold.
- URLs that are not http or https (the app's own pages, `data:`, `blob:`) go
  to the webview's `fetch`.

[The documentation](https://github.com/egoist/mygo/blob/main/docs/plugins/fetch.md)
covers streaming, errors and the options of the Go side.
