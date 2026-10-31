# Loopback end to end

`tools/loopback-e2e.sh` writes these files. `summary.md` names the source commit, Xcode and the
simulator; `local-api.txt` and `design-tour.txt` list each UI test with its result and seconds, then
the totals over the whole run, and the text Voxt returned for `testdata/jfk.wav`.

- `local-api.txt`: Try it and a dictation into the app's own text field, through the Voxt keyboard,
  against Voxt's Local API on this Mac with the models selected in Voxt.
- `design-tour.txt`: the design tour against `tools/design-tour-fixture.py`, which answers with the
  recorded contract and no model.

A freshly erased simulator differs from a used one in five ways, and the tests handle each:

- Settings lists the Voxt keyboard only after Voxt has been opened once, so the tests open it before
  turning the keyboard on.
- The first keyboard change shows iOS's one-time "Quickly Change Keyboards" tip over the keyboard;
  the tests close it with Continue.
- Safari's first load of the fixture's page can stay blank or keep loading; the tests reload it or
  open it again.
- Safari's one-time toolbar tip takes the next tap; the tests close it before tapping the field.
- The first system keyboard shows iOS's one-time slide-to-type tip over the keys, and switching
  keyboards does nothing until it is closed; the tests close it with Continue.

Run with `--erase` to start from that state.
