Tree: feat/voxt-mobile at 80e4b5090f00ad1a1a0bab854f2cbe6bc817c63d; the commits after it on the submitted branch only gate a Keychain test and rebase the series onto main.

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
