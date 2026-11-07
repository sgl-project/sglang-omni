Responses of OpenAI's POST /v1/audio/transcriptions, written from its API
reference (response_format json and verbose_json, and its error bodies)
and Groq's compatible answers; the texts are made up. The .golden files
are what the client sent, as openai_test.go prints a request.
error-400-no-speech-stepfun.json is StepFun's answer (api.stepfun.com,
step-asr, 2026-10-15) to a second of silence: nothing was said, which the
client reads as an empty text.
