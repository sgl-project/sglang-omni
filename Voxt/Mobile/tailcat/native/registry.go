package main

import "sync"

type bridgeSession struct {
	forward *Forward
	close   func()
	once    sync.Once
}

func (s *bridgeSession) Close() {
	if s == nil {
		return
	}
	s.once.Do(func() {
		if s.forward != nil {
			s.forward.Close()
		}
		if s.close != nil {
			s.close()
		}
	})
}

var (
	registryMu sync.Mutex
	nextHandle int64 = 1
	sessions         = map[int64]*bridgeSession{}
)

func register(session *bridgeSession) int64 {
	registryMu.Lock()
	defer registryMu.Unlock()
	handle := nextHandle
	nextHandle++
	sessions[handle] = session
	return handle
}

func Stop(handle int64) int {
	registryMu.Lock()
	session := sessions[handle]
	if session != nil {
		delete(sessions, handle)
	}
	registryMu.Unlock()
	if session == nil {
		return 0
	}
	session.Close()
	return 0
}
