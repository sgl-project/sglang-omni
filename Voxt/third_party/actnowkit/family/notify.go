package family

import "sync"

// Topics of change notifications (spec §5.4). The notification name is
// NotifyPrefix + topic: "dev.actnow.family.settings".
const (
	TopicApps        = "apps"        // apps/*.json written
	TopicSettings    = "settings"    // settings.json written
	TopicCredentials = "credentials" // credentials.json or a shared Keychain item written
	TopicModels      = "models"      // models/** written, or a download / delete under $HUB finished
	TopicRuntime     = "runtime"     // runtime/*.json written
	TopicMemory      = "memory"      // asks other apps to unload idle models (spec §11.1)
	TopicAccount     = "account"     // account/ written (spec §12)
)

// NotifyPrefix starts every family notification name.
const NotifyPrefix = "dev.actnow.family."

// NotifyName is the notification name of topic for this directory: the
// spec's "dev.actnow.family.<topic>" for the real family directory, and
// "dev.actnow.family.<hash of the root>.<topic>" for any other (tests,
// ACTNOW_FAMILY_DIR), so that test runs never wake the user's apps.
func (d Dir) NotifyName(topic string) string {
	if d.isCanonical() {
		return NotifyPrefix + topic
	}
	return NotifyPrefix + d.id() + "." + topic
}

// Post sends the change notification of topic (notify_post on macOS).
// Notifications are hints: errors are returned for logging only, and
// readers re-read on activation anyway (spec §5.4).
func (d Dir) Post(topic string) error {
	name := d.NotifyName(topic)
	localPost(name)
	return post(name)
}

// Watcher receives change notifications. C carries topics; deliveries are
// dropped when the receiver is 16 behind, which only delays a re-read.
type Watcher struct {
	C <-chan string

	c      chan string
	topics map[string]string // notification name → topic
	mu     sync.Mutex
	closed bool
	stop   func()
}

// Watch subscribes to topics (notify_register_file_descriptor on macOS;
// elsewhere only Posts of this process are seen). Close the watcher when
// done.
func (d Dir) Watch(topics ...string) (*Watcher, error) {
	w := &Watcher{c: make(chan string, 16), topics: map[string]string{}}
	w.C = w.c
	names := make([]string, 0, len(topics))
	for _, t := range topics {
		n := d.NotifyName(t)
		w.topics[n] = t
		names = append(names, n)
	}
	stopLocal := localWatch(names, w.deliver)
	stopSys, err := watch(names, w.deliver)
	if err != nil {
		stopLocal()
		return nil, err
	}
	w.stop = func() { stopLocal(); stopSys() }
	return w, nil
}

func (w *Watcher) deliver(name string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return
	}
	t, ok := w.topics[name]
	if !ok {
		return
	}
	select {
	case w.c <- t:
	default:
	}
}

// Close stops the watcher and closes C.
func (w *Watcher) Close() error {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return nil
	}
	w.closed = true
	w.mu.Unlock()
	w.stop()
	w.mu.Lock()
	close(w.c)
	w.mu.Unlock()
	return nil
}

// In-process delivery, for systems without Darwin notifications (and to
// keep tests on Linux meaningful). On macOS notify_post reaches this
// process's own registrations, so local delivery is off there.

var local struct {
	sync.Mutex
	next int
	subs map[int]localSub
}

type localSub struct {
	names   []string
	deliver func(string)
}

func localWatch(names []string, deliver func(string)) func() {
	if systemNotify {
		return func() {}
	}
	local.Lock()
	defer local.Unlock()
	if local.subs == nil {
		local.subs = map[int]localSub{}
	}
	local.next++
	id := local.next
	local.subs[id] = localSub{names: names, deliver: deliver}
	return func() {
		local.Lock()
		delete(local.subs, id)
		local.Unlock()
	}
}

func localPost(name string) {
	if systemNotify {
		return
	}
	local.Lock()
	var fns []func(string)
	for _, s := range local.subs {
		for _, n := range s.names {
			if n == name {
				fns = append(fns, s.deliver)
			}
		}
	}
	local.Unlock()
	for _, fn := range fns {
		fn(name)
	}
}
