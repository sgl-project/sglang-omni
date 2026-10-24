Tree: this change on upstream main.

Host: XCTest host (LocalAPILoopbackServeTests). 11 of 11 checks passed.

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
| dictation-after-idle | pass | after both models unloaded on idle, the dictation answers within 90 s with the text of jfk.wav |
