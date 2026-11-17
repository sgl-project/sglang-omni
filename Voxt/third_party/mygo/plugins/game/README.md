# game

What games need beyond what apps do, for [MyGo](https://github.com/egoist/mygo),
opt-in: MyGo is an app framework.

- `game.View` runs a game's loop in a native UI element, in fixed steps
  with the player's input, `game.Camera` looks at its world, and
  `game.IntegerFit` shows a canvas of pixel art pixel-perfect at any size.
- `gamepad` reads game controllers in one standard layout, on macOS
  (GameController.framework), Windows (XInput), Linux (evdev) and in
  browsers (the Gamepad API).

```go
import "github.com/egoist/mygo/plugins/game/gamepad"

for _, pad := range gamepad.Pads() {
	s := pad.State()
	x, y := s.LeftStick()
	move(x, y)
}
```

See [the documentation](../../docs/plugins/game.md).
