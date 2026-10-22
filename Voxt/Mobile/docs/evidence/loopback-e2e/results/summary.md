Tree: feat/voxt-mobile at 0ff031b42983f25ece44ad2403b0171815bdab0c.

Host: XCTest host (LocalAPILoopbackServeTests). 10 of 10 checks passed.

| Check | Result | Compared |
| --- | --- | --- |
| health | pass | keys, app and ok as in health.json |
| capabilities | pass | equal to capabilities.json apart from version |
| dictation-jfk | pass | keys of dictation-response.json; the text has 'country' |
| dictation-zh | pass | the text is in Chinese characters |
| compose-translate | pass | keys of compose-response.json; jfk.wav translated to zh-Hans |
| compose-prompts | pass | one built-in prompt per mode |
| correction | pass | shape of correction-response.json (the pair, or null) |
| error-unauthorized | pass | equal to errors.json |
| error-notFound | pass | equal to errors.json |
| error-wrongMethod | pass | equal to errors.json |
