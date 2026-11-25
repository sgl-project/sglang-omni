package icons

import _ "embed"

//go:embed voxt-microphone.svg
var voxtMicrophoneSVG string

// VoxtMicrophone is the brand's microphone without its tile or stars.
var VoxtMicrophone = voxt("microphone", voxtMicrophoneSVG)
