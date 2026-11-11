//go:build windows

package permission

// Native is the system's permissions. Windows has no macOS-style TCC
// gate: waveIn capture and clipboard/send-input text insertion are always
// available (unless the user disables desktop-app microphone access in
// Privacy settings), so every permission reads granted.
func Native() System { return windowsSys{} }

type windowsSys struct{}

func (windowsSys) Microphone() State { return Granted }
func (windowsSys) RequestMicrophone(done func(bool)) {
	go done(true)
}
func (windowsSys) AccessibilityTrusted() bool { return true }
func (windowsSys) PromptAccessibility()        {}
func (windowsSys) SpeechRecognition() State    { return Granted }
func (windowsSys) RequestSpeechRecognition(done func(bool)) {
	go done(true)
}
