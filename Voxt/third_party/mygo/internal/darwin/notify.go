//go:build darwin

package darwin

import (
	"sync"

	"github.com/ebitengine/purego/objc"

	"github.com/egoist/mygo/internal/platform"
)

// Notifications use UNUserNotificationCenter, which only works for apps
// running from a bundle with an identifier.

var (
	mainQueueMu sync.Mutex
	mainQueue   []func()
)

// runOnMain runs fn on the main thread; used by callbacks the system
// delivers on background queues.
func (b *Backend) runOnMain(fn func()) {
	if b.IsMainThread() {
		fn()
		return
	}
	b.post(fn)
}

// post runs fn on the main thread after the current event, even when
// called there.
func (b *Backend) post(fn func()) {
	mainQueueMu.Lock()
	mainQueue = append(mainQueue, fn)
	mainQueueMu.Unlock()
	b.Signal()
}

func drainMainQueue() {
	mainQueueMu.Lock()
	q := mainQueue
	mainQueue = nil
	mainQueueMu.Unlock()
	for _, fn := range q {
		fn()
	}
}

func registerNotificationDelegate() {
	if !hasClass("UNUserNotificationCenter") {
		return
	}
	classDef("MyGoNotificationDelegate", "NSObject", []string{"UNUserNotificationCenterDelegate"}, []objc.MethodDef{
		method("userNotificationCenter:willPresentNotification:withCompletionHandler:", func(self id, _ objc.SEL, center, n id, handler uintptr) {
			// Show banners even while the app is in the foreground.
			callBlock(handler, 1<<1|1<<3|1<<4) // sound | list | banner
		}),
		method("userNotificationCenter:didReceiveNotificationResponse:withCompletionHandler:", func(self id, _ objc.SEL, center, resp id, handler uintptr) {
			ident := goString(send(send(send(send(resp, "notification"), "request"), "identifier"), "self"))
			theBackend.runOnMain(func() { theBackend.h.NotificationClicked(ident) })
			callBlock(handler)
		}),
	})
}

func (b *Backend) NotificationsSupported() bool {
	_, packaged := appController{b}.Package()
	return packaged && hasClass("UNUserNotificationCenter")
}

// notificationCenter returns the app's notification center. It throws
// when the app does not run from a bundle: check NotificationsSupported
// first.
func notificationCenter() id {
	return send(class("UNUserNotificationCenter"), "currentNotificationCenter")
}

// attachNotificationDelegate gives the notification center the delegate
// that receives clicks, at launch: the system delivers the click that
// launched the app before the app has finished launching, and drops it
// when the center has no delegate yet.
func (b *Backend) attachNotificationDelegate() {
	if !b.NotificationsSupported() {
		return
	}
	b.notifyDelegate = alloc("MyGoNotificationDelegate")
	withPool(func() {
		send(notificationCenter(), "setDelegate:", uintptr(b.notifyDelegate))
	})
}

// waitingNotification is a notification shown before the user has answered
// whether the app may show notifications.
type waitingNotification struct {
	n    *platform.Notification
	done func(error)
}

// notifyOptions are the permissions asked for: badge, sound and alert.
const notifyOptions = 1<<0 | 1<<1 | 1<<2

func (b *Backend) ShowNotification(n *platform.Notification, done func(error)) {
	if !b.NotificationsSupported() {
		done(platform.ErrUnsupported)
		return
	}
	if b.notifyAnswered {
		b.addNotification(n, done)
		return
	}
	// The system drops what it is given before the user has answered
	// whether the app may show notifications, so the first notification
	// asks and the ones shown meanwhile wait for the answer. The system
	// only prompts the first time; afterwards it answers at once, and
	// adding a notification fails when System Settings does not allow it.
	b.notifyWaiting = append(b.notifyWaiting, waitingNotification{n, done})
	if len(b.notifyWaiting) > 1 {
		return
	}
	withPool(func() {
		blk := newBlock(func(_ objc.Block, granted bool, err id) {
			b.runOnMain(func() {
				b.notifyAnswered = true
				waiting := b.notifyWaiting
				b.notifyWaiting = nil
				for _, w := range waiting {
					b.addNotification(w.n, w.done)
				}
			})
		})
		send(notificationCenter(), "requestAuthorizationWithOptions:completionHandler:",
			notifyOptions, uintptr(blk))
		blk.Release()
	})
}

// addNotification gives the system a request to show a notification. Its
// trigger is nil, which shows it at once.
func (b *Backend) addNotification(n *platform.Notification, done func(error)) {
	withPool(func() {
		content := autorelease(alloc("UNMutableNotificationContent"))
		send(content, "setTitle:", uintptr(nsString(n.Title)))
		if n.Subtitle != "" {
			send(content, "setSubtitle:", uintptr(nsString(n.Subtitle)))
		}
		send(content, "setBody:", uintptr(nsString(n.Body)))
		if !n.Silent {
			send(content, "setSound:", uintptr(send(class("UNNotificationSound"), "defaultSound")))
		}
		if n.Group != "" {
			send(content, "setThreadIdentifier:", uintptr(nsString(n.Group)))
		}
		req := send(class("UNNotificationRequest"), "requestWithIdentifier:content:trigger:",
			uintptr(nsString(n.ID)), uintptr(content), 0)
		blk := newBlock(func(_ objc.Block, nsErr id) {
			// The error is the caller's only while the block runs.
			var err error
			withPool(func() { err = notificationError(nsErr) })
			b.runOnMain(func() { done(err) })
		})
		send(notificationCenter(), "addNotificationRequest:withCompletionHandler:", uintptr(req), uintptr(blk))
		blk.Release()
	})
}

// unErrorNotificationsNotAllowed is UNErrorCodeNotificationsNotAllowed.
const unErrorNotificationsNotAllowed = 1

// notificationError describes an error of the notification center.
func notificationError(err id) error {
	if err == 0 {
		return nil
	}
	if goString(send(err, "domain")) == "UNErrorDomain" && sendInt(err, "code") == unErrorNotificationsNotAllowed {
		return platform.ErrNotificationsDenied
	}
	return nsError(err)
}

func (b *Backend) RemoveNotification(ident string) {
	if !b.NotificationsSupported() {
		return
	}
	// One still waiting for the user's answer is not shown.
	waiting := b.notifyWaiting[:0]
	for _, w := range b.notifyWaiting {
		if w.n.ID == ident {
			w.done(nil)
		} else {
			waiting = append(waiting, w)
		}
	}
	b.notifyWaiting = waiting
	withPool(func() {
		center := notificationCenter()
		ids := nsArray(nsString(ident))
		send(center, "removeDeliveredNotificationsWithIdentifiers:", uintptr(ids))
		send(center, "removePendingNotificationRequestsWithIdentifiers:", uintptr(ids))
	})
}

func (b *Backend) RemoveAllNotifications() {
	if !b.NotificationsSupported() {
		return
	}
	waiting := b.notifyWaiting
	b.notifyWaiting = nil
	for _, w := range waiting {
		w.done(nil)
	}
	withPool(func() {
		center := notificationCenter()
		send(center, "removeAllDeliveredNotifications")
		send(center, "removeAllPendingNotificationRequests")
	})
}
