package session

// Model names a model a session used.
type Model struct {
	ID, Name, Model string
	// Remote: its requests can run side by side (local.Model.Remote).
	Remote bool
}
