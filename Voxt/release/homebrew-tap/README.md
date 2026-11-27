# Homebrew tap for Voxt

[Voxt](https://github.com/sgl-project/sglang-omni/tree/main/Voxt) is a voice input, translation and rewriting app for Macs with Apple silicon.

```sh
brew install --cask sgl-project/voxt/voxt
```

Voxt updates itself after it is installed, so `brew upgrade` leaves it alone unless you pass `--greedy`.

## How this tap is kept current

- `Casks/voxt.rb` is written by `Voxt/scripts/homebrew-cask.sh` in sgl-project/sglang-omni. Change the cask there, where its tests run, not only here.
- The Bump Voxt workflow checks sgl-project/sglang-omni every hour. When a newer stable `voxt-vX.Y.Z` release is published, it writes the cask, runs `brew style` and `brew audit`, and opens a pull request. Merge it. Betas are never offered.
- Cask checks runs the same checks on every other pull request.
- No secret is needed. The repository setting "Allow GitHub Actions to create and approve pull requests" must stay on.

The workflows here are copies of `Voxt/release/homebrew-tap` in sgl-project/sglang-omni. Change them there first, then copy them here.
