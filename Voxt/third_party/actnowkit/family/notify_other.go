//go:build !darwin

package family

// systemNotify is false: notifications stay inside the process.
const systemNotify = false

func post(string) error { return nil }

func watch([]string, func(string)) (func(), error) { return func() {}, nil }
