package ui

import (
	"slices"
	"sync"
	"time"
	"weak"

	"github.com/egoist/mygo/i18n"
	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/internal/scene"
	"github.com/egoist/mygo/internal/text"
	"github.com/egoist/mygo/transfer"
)

// host is where a runtime's frames go: a window's surface, or memory for
// headless rendering and tests.
type host interface {
	size() (w, h, scale float32)
	// refreshRate returns how many times a second the display refreshes,
	// 0 when unknown.
	refreshRate() float32
	// occluded reports whether nothing of the window shows, as when other
	// windows cover it.
	occluded() bool
	present(s *scene.Scene)
	requestFrame()
	// afterTimer and repaintTimer arm the timers of After and
	// Painter.After for at, or stop them for the zero time.
	afterTimer(at time.Time)
	repaintTimer(at time.Time)
	setCursor(Cursor)
	setTextInput(t platform.TextInputState)
	updateAccessibility(tree *platform.AccessTree)
	readClipboard() string
	writeClipboard(string)
	// readClipboardData reads the representations of formats on the
	// clipboard, all for nil; writeClipboardData puts data there;
	// clipboardFormats lists them.
	readClipboardData(formats []transfer.Format) transfer.Data
	clipboardFormats() []transfer.Format
	writeClipboardData(transfer.Data)
	startDrag()
	startDataDrag(transfer.Data, any, transfer.DragOptions, float32, float32) error
	cancelDataDrag()
	setDropFormats([]transfer.Format)
	titleBarDoubleClicked()
	// escape tells the window an Escape nothing handled.
	escape()
	isDark() bool
	preferences() platform.Preferences
	// locale returns the language the app shows, and region the user's
	// regional formats; they cost nothing.
	locale() string
	region() i18n.Region
	// rightToLeft reports whether the window is laid out right to left,
	// as its language reads.
	rightToLeft() bool
	titleBar() TitleBar
	// vibrancy reports whether the window shows a material where the
	// frames are transparent.
	vibrancy() bool
	// invalidate asks for a frame from any goroutine.
	invalidate()
	// post runs fn on the main thread soon; it is safe from any goroutine.
	post(fn func())
	openURL(url string, done func(error))
	// popupMenu shows a context menu at (x, y) after the event being
	// handled; chosen receives the ID of the item chosen.
	popupMenu(m *platform.Menu, x, y float32, chosen func(id int))
	// placeWebViews shows the web views where the frame shows them, and
	// hides the others.
	placeWebViews(views []placedWebView)
	// tabIntoWebView gives the page of v the keyboard as Tab does, from
	// its end when back (platform.WebView.TabInto).
	tabIntoWebView(v NativeWebView, back bool)
	// platformWebView returns the web view of the platform v is, if any.
	platformWebView(v NativeWebView) platform.WebView
}

// engine runs the user interface of one window: it builds frames with
// the view function, lays them out, paints them and routes input to the
// elements of the last frame. Main thread only, except where noted.
type engine struct {
	// inlineRects is placeInline's scratch, kept from frame to frame.
	inlineRects []text.Rect
	// byPosition holds the keys whose press stood for its physical key
	// (keyDownPhysical), until their release does the same.
	byPosition      map[Key]bool
	owner           weak.Pointer[engine]
	epoch           uint64
	parts           []any
	public          *Context
	arena           *elementOwner
	handleChecks    bool
	afterInputs     []inputAction
	inputs          []valueInput
	inputAt         int
	afterInputAt    int
	applyingInputs  bool
	notices         []*state
	typedInputs     []*state
	focusFields     map[any]*focusField
	refs            []*refState
	actions         []action
	textInputClosed bool
	view            func(*context)
	host            host
	c               context
	text            *text.System
	scene           scene.Scene
	painter         Painter
	glyphRun        glyphRun

	// building counts the public constructors under way: input queued
	// while more were under way than now belongs to controls built since,
	// which applyInputs finalizes (see value_input.go).
	building int32
	// waitingInputs and waitingActions are the entries of inputs and
	// afterInputs whose containers are still building.
	waitingInputs, waitingActions []int

	// unsmoothed draws the window's text without font smoothing
	// (Content.FontSmoothing).
	unsmoothed bool
	// measured are the last spans laid out outside elements (richParams).
	measured     [8]measuredSpans
	nextMeasured int
	paths        paths
	svgs         svgs
	flex         flexScratch
	grid         gridScratch
	// under is the painter's stack of opaque backgrounds, and holes the
	// holes it painted, kept from one frame to the next.
	under []opaqueBox
	holes []Rect

	states map[uint64]*state
	// free are states pruned, which new elements take: rows coming into
	// a list's view take those of rows that went out of it.
	free  []*state
	frame uint64
	// pass is the pass of the view building the frame: the last one
	// builds the elements that stay.
	pass int

	// What the last frame laid out, for input until the next one: the
	// focus order, with the scope of each element, and the dialog on top.
	hits        []hit
	focusOrder  []uint64
	focusScopes []focusScope
	modal       uint64
	// modalLayer is the element at the top of the overlay holding the
	// dialog on top, which assistive technology sees with what is above.
	modalLayer  uint64
	commitScope focusScope
	// webPlaces are where the frame shows web views (WebView), and webOut
	// what the host got of them; webShown tells that the last frame
	// showed some.
	webPlaces []webPlace
	webOut    []placedWebView
	webShown  bool
	// The focus groups of the frame, the group of each element of the
	// focus order in one, and the element of each that had the focus
	// last.
	// clickLater are the elements the view clicked, as a toolbar's
	// overflow menu does, whose clicks the next pass sees.
	clickLater []uint64
	groups     map[uint64]groupInfo
	memberOf   map[uint64]uint64
	groupLast  map[uint64]uint64
	// openers are the elements that had the focus as overlays opened, by
	// overlay; downs the elements the pointer went down on since the last
	// pass (0 for none), for PressedOutside.
	openers   map[uint64]uint64
	downs     []uint64
	regs      []shortcutReg
	nextRegs  []shortcutReg
	delivered []shortcutReg

	// trans are the transitions of the elements given one, by ID
	// (transition.go): exitsBuilt tells that the last frame built elements
	// with an Exit, whose arena the next frame keeps to copy those that
	// go, laidW and laidH are the size the last frame was laid out at, and
	// byID maps a frame's elements, for the copies going.
	trans map[uint64]*transition
	// matched is where the elements of a Transition's Match showed.
	matched map[any]*matchRec
	// instantFrame is the frame that DisableAnimations shows at once.
	instantFrame uint64
	// paints counts the frames painted, those painting the last frame's
	// elements again included.
	paints uint64
	// posed are the elements of the last frame built with a pose
	// (AnimatePose), parents first.
	posed        []*node
	exitsBuilt   bool
	laidW, laidH float32
	byID         map[uint64]*node
	// insp is the inspector (inspector.go); dupKeys are the duplicate keys
	// reported, and warnings what the inspector lists.
	insp inspector
	// dirOverride lays the window out left to right (1) or right to left
	// (2) whatever its language, as the inspector's toggle asks; 0 follows
	// it.
	dirOverride uint8
	dupKeys     map[uint64]bool
	warnings    []string
	// strict makes mistakes found while building panic, as duplicate keys
	// do in a Tester.
	strict bool
	// clock is a Tester's time, time.Now when nil.
	clock func() time.Time

	pointerX, pointerY float32
	pointerIn          bool
	hover              []uint64
	// chain is a buffer for the elements under the pointer.
	chain       []uint64
	pressed     *state
	pressButton int
	// pressClicks is the press's count of clicks, which its release
	// carries.
	pressClicks int
	// pressDragger is the element around the one pressed that drags a
	// value, which takes the press as it moves (dragMove), or 0.
	pressDragger uint64
	focused      uint64
	// texts are selectable paragraphs in build order; selection spans those
	// of one Selectable container (textselection.go).
	texts        []*state
	selection    textSelection
	focusVisible bool
	// keys are the keys pressed since the last frame, for its shortcuts;
	// retried, keys no shortcut of the frame they came to took, which the
	// shortcuts that frame built take in the next one; unrouted, those
	// the frame being built gave no shortcut, or gave one the view no
	// longer asks for (routeKeys, settleKeys).
	keys, retried []keyEvent
	unrouted      []shortcutReg
	menu          menuState
	// windowFocused tells that the window has the keyboard focus, which
	// windowFocus reads; focusRead, that the last frame read it, so that a
	// change of it draws a frame.
	windowFocused, focusRead bool
	// toasts are the toasts the window holds, oldest first, and
	// toastList those showing, as ToastViewportBase gives them; their time
	// stops while toastsPaused. A viewport of them was built in the pass
	// toastPass of the frame toastFrame.
	toasts       []toast
	nextToast    uint64
	toastList    []Toast
	toastsPaused bool
	toastFrame   uint64
	toastPass    int
	// mods are the modifiers of the last pointer event.
	mods Modifiers

	consumed bool
	// animating is set as a frame builds when something moves, for another
	// frame built anew; repainting as it paints when only drawings move
	// (Painter.AnimationFrame), for a frame painting its elements again,
	// and repaintAt to when drawings change next (Painter.After). redraw
	// tells that the next frame may paint again, as no event came and
	// nothing else asked for one since; painted is the size and scale of
	// the window, and gen the text system's Generation, as the last frame
	// was built. The host's repaintTimer asks for a frame at repaintDue,
	// unless one came or was asked for since. held tells that what moves
	// waits for the window to show.
	animating  bool
	repainting bool
	repaintAt  time.Time
	redraw     bool
	held       bool
	painted    [3]float32
	gen        uint64
	repaintDue time.Time
	// late is set when lists built elements while laying out.
	late bool
	// revealIDs are the elements to scroll into view once the frame is
	// laid out.
	revealIDs []uint64
	wakeMu    sync.Mutex
	wakeAt    time.Time
	cursor    Cursor
	// ime is the text input state the host has, and base the rune of the
	// focused editor its text starts at.
	ime struct {
		state platform.TextInputState
		base  int
	}
	// drag is the value being dragged within the window.
	drag        *valueDrag
	incoming    *platform.DataDragEvent
	dataOver    uint64
	closed      bool
	closing     bool // the window closed during a frame, which closes the engine as it ends
	dropFormats []transfer.Format
	dropScratch []transfer.Format
	// kept are the pages of the history that Routers keep, and commitPage
	// the page around the elements being committed.
	kept       map[uint64]bool
	commitPage uint64
	// announcements are the texts for assistive technology to read out
	// (Context.Announce).
	announcements []string
	// stats measures frames for MYGO_FRAME_STATS, nil when it is unset.
	stats *frameStats
	// dropOver is the element files are dragged over; access is true once
	// assistive technology asked for the content.
	dropOver   uint64
	access     bool
	blinkStart time.Time
	inFrame    bool
	dark       bool
	darkKnown  bool
	// prefs are the desktop's preferences, read once until they change.
	prefs      Preferences
	prefsKnown bool
	// lang is the language the app shows, read as each frame starts.
	lang string
	// fmtRegion and fmtLocale are the formats the controls write dates
	// in, worked out of the language and the system's formats, fmtSys
	// (context.formats); fmtCache keeps what they wrote.
	fmtKnown  bool
	fmtLang   string
	fmtSys    i18n.Region
	fmtRegion i18n.Region
	fmtLocale string
	fmtCache  map[fmtKey]string
	// theme is the default theme, which follows the appearance and the
	// preferences, made once until they change (themeOK); each pass
	// starts from a copy, passTheme, which the view may change.
	theme      Theme
	themeOK    bool
	passTheme  Theme
	collect    bool
	labels     []labelNode
	tips       tooltips
	scrollDrag scrollDrag
	// lastPress is the element of the press (id) and of the one before
	// (prev), and how many of the backend's count of clicks were before
	// the first on it.
	lastPress struct {
		id, prev uint64
		base     int
	}
}

// scrollDrag is the scroll bar thumb being dragged: where the pointer
// and the offset started, and the size of the content then, which the
// thumb keeps until it is let go, as the rows of a List measured
// meanwhile change it.
type scrollDrag struct {
	st                 *state
	start              float32
	from               float64
	contentW, contentH float64
	horizontal         bool
}

// labelNode is an element showing or labeled with text, for tests and
// assistive technology.
type labelNode struct {
	id   uint64
	text string
	r    Rect
}

type hit struct {
	st    *state
	r     Rect
	flags uint32
}

type shortcutReg struct {
	id   uint64
	mods Modifiers
	key  Key
	// overlay marks the registration of an overlay (overlayShortcut),
	// made serial-th in its pass: the overlay made last is on top.
	overlay bool
	serial  int32
	// retried marks a key delivered, or left unrouted, in the frame after
	// the one it came to (settleKeys): it is not retried again.
	retried bool
}

type keyEvent struct {
	mods Modifiers
	key  Key
}

func newRuntime(view func(*context), h host) *engine {
	rt := &engine{view: view, host: h, text: textSystem(), states: map[uint64]*state{}, windowFocused: true}
	rt.c.rt = rt
	rt.owner = weak.Make(rt)
	rt.arena = &elementOwner{rt: rt}
	rt.public = &Context{rt: rt, services: Services{owner: rt.owner}}
	rt.handleChecks = developmentHandles
	if frameStatsOn {
		rt.stats = newFrameStats(frameStatsThreshold)
	}
	return rt
}

func (rt *engine) defaultTheme() *Theme {
	if !rt.darkKnown {
		rt.dark, rt.darkKnown = rt.host.isDark(), true
		rt.themeOK = false
	}
	if !rt.prefsKnown {
		rt.themeOK = false
	}
	if !rt.themeOK {
		t := LightTheme()
		if rt.dark {
			t = DarkTheme()
		}
		t.follow(rt.preferences())
		rt.theme, rt.themeOK = *t, true
	}
	rt.passTheme = rt.theme
	return &rt.passTheme
}

// themeChanged follows a change of the system appearance, or of the
// desktop's preferences.
func (rt *engine) themeChanged() {
	rt.darkKnown, rt.prefsKnown = false, false
	rt.redraw, rt.repaintDue = false, time.Time{}
	rt.host.requestFrame()
}

// runFrame builds, lays out, paints and presents a frame.
func (rt *engine) runFrame() {
	if rt.inFrame || rt.closed {
		return
	}
	rt.inFrame = true
	defer rt.endFrame()

	rt.frame++
	rt.focusRead = false
	rt.stats.begin(rt)
	now := rt.now()
	w, h, scale := rt.host.size()
	// The content takes the room the inspector leaves.
	appW := rt.insp.contentWidth(w)
	rt.insp.lap(-1)
	rt.c.titleBar = rt.host.titleBar()
	rt.c.vibrancy = rt.host.vibrancy()
	rt.text.BeginFrame()
	rt.gen = rt.text.Generation()
	rt.painted = [3]float32{w, h, scale}
	rt.animating, rt.repainting, rt.repaintAt = false, false, time.Time{}
	rt.routeKeys()
	if rt.drag != nil {
		// The source's element is this frame's, if it builds one.
		rt.drag.elem = nil
		rt.dragScroll()
	}
	rt.scrollTextSelection()

	if rt.exitsBuilt {
		// The last frame's elements stay as they are while this one builds,
		// for copies of those that go with an exit transition.
		rt.c.chunks, rt.c.spare = rt.c.spare, rt.c.chunks
		rt.c.dirty, rt.c.spareDirty = rt.c.spareDirty, rt.c.dirty
		rt.c.used = rt.c.dirty
	}

	// An event handled while building (a click, an edit) may change what
	// was built before it: build again, so the frame shows the outcome.
	for pass := 0; pass < 3; pass++ {
		rt.pass = pass
		rt.consumed = false
		rt.nextRegs = rt.nextRegs[:0]
		clear(rt.kept)
		rt.c.reset(now, appW, h)
		rt.view(&rt.c)
		if rt.closing {
			// The view closed the window, as a close button does: the
			// window is gone at once, and the frame ends with the view.
			return
		}
		rt.buildToasts(&rt.c)
		if ov := rt.c.overlay; ov != nil {
			rt.c.root.add(ov)
		}
		if rt.insp.open {
			rt.buildInspector(&rt.c, appW, w, h)
		}
		rt.applyAllInputs()
		rt.finishInputs()
		// Commit derived values while callbacks still hold this pass's
		// bindings. Delivered callbacks do not repeat in the next pass.
		rt.runNoticeActions()
		rt.applyFocusRequests()
		rt.runActions()
		rt.prepareSelectable(rt.c.root)
		rt.resolveMenu()
		rt.endPass()
		if rt.closing {
			return // an action closed it
		}
		if !rt.consumed {
			break
		}
	}
	if rt.stats != nil {
		rt.stats.passes = rt.pass + 1
	}
	rt.stats.lap(phaseBuild)
	rt.insp.lap(0)
	root := rt.c.root
	layoutTree(root, appW, h)
	rt.commit(root, w, h)
	rt.commitFocusBindings()
	clear(rt.texts)
	rt.texts = rt.texts[:0]
	rt.collectSelectable(root, false)
	rt.syncTextSelection()
	rt.stats.lap(phaseLayout)
	rt.insp.lap(1)
	if rt.insp.open {
		rt.insp.snapshot(rt, root)
	}
	rt.paint(root, w, h, scale)
	for try := 0; try < 2 && rt.text.Full(); try++ {
		// The glyph atlas filled up and left some out: make room, keeping
		// what the frame draws, and paint it again.
		rt.text.MakeRoom()
		rt.paint(root, w, h, scale)
	}
	rt.stats.lap(phasePaint)
	rt.insp.lap(2)
	rt.host.present(&rt.scene)
	rt.placeWebViews()
	rt.stats.lap(phasePresent)
	rt.prune()
	rt.syncDropFormats()
	rt.prunePictures()
	rt.text.EndFrame()
	rt.settleKeys()
	rt.regs, rt.nextRegs = rt.nextRegs, rt.regs
	rt.updateTextInput()
	rt.updateCursor()
	if rt.access {
		rt.host.updateAccessibility(rt.accessTree())
	}
	if h, ok := rt.host.(*headless); ok {
		// For tests, whether assistive technology reads the window or not.
		h.announced = append(h.announced, rt.announcements...)
	}
	rt.announcements = rt.announcements[:0]
	rt.next()
	if len(rt.retried) > 0 {
		// The keys retried route in a frame built anew, at once.
		rt.redraw, rt.repaintDue = false, time.Time{}
		rt.host.requestFrame()
	}
	rt.armTimer()
	rt.showMenu()
	rt.c.finish()
	clear(rt.byID)
	rt.stats.end(rt)
}

// now returns the time, which a Tester sets (clock).
func (rt *engine) now() time.Time {
	if rt.clock != nil {
		return rt.clock()
	}
	return time.Now()
}

// next asks for the frame that what moves needs: one built anew while the
// view animates, one painting the elements again while only drawings move,
// as soon as the display can show it or when they change next. While
// nothing of the window shows, what moves waits until some of it does
// (SurfaceShown) rather than draw frames nobody sees, as browsers pause
// the animation frames of windows out of sight.
func (rt *engine) next() {
	rt.redraw, rt.repaintDue, rt.held = false, time.Time{}, false
	moving := rt.animating || rt.repainting || !rt.repaintAt.IsZero()
	switch {
	case moving && rt.host.occluded():
		rt.held = true
	case rt.animating:
		rt.host.requestFrame()
	case rt.repainting:
		rt.redraw = true
		rt.host.requestFrame()
	case !rt.repaintAt.IsZero():
		rt.redraw, rt.repaintDue = true, rt.repaintAt
	}
	rt.host.repaintTimer(rt.repaintDue)
}

// repaintNow asks for the frame that Painter.After asked for, unless
// another frame came, or was asked for, since. An event since that asked
// for none, as the pointer moving over elements that do not look at it,
// leaves it due: it builds the view anew, in case the event changed what
// the view shows.
func (rt *engine) repaintNow() {
	if !rt.repaintDue.IsZero() && !rt.now().Before(rt.repaintDue.Add(-time.Millisecond)) {
		rt.repaintDue = time.Time{}
		rt.host.requestFrame()
	}
}

// surfaceFrame draws the frame the surface asked for: the last frame's
// elements painted again when only drawings moved since, else a frame
// built anew.
func (rt *engine) surfaceFrame() {
	if w, h, scale := rt.host.size(); rt.redraw && !rt.inFrame && rt.c.root != nil &&
		rt.painted == [3]float32{w, h, scale} && rt.text.Generation() == rt.gen {
		rt.repaintFrame(w, h, scale)
		return
	}
	rt.runFrame()
}

// repaintFrame paints the elements of the last frame again, at the time of
// this one, for drawings that move with it (Painter.AnimationFrame) while
// nothing else changed: the view is neither built nor laid out, so what
// moves costs only its painting. Its timers stay as the last frame built
// armed them.
func (rt *engine) repaintFrame(w, h, scale float32) {
	rt.inFrame = true
	defer rt.endFrame()
	rt.stats.begin(rt)
	start := time.Now()
	rt.c.now = rt.now()
	rt.text.BeginFrame()
	rt.repainting, rt.repaintAt = false, time.Time{}
	root := rt.c.root
	rt.repose()
	rt.paint(root, w, h, scale)
	for try := 0; try < 2 && rt.text.Full(); try++ {
		rt.text.MakeRoom()
		rt.paint(root, w, h, scale)
	}
	rt.stats.lap(phasePaint)
	rt.insp.repainted(time.Since(start))
	rt.host.present(&rt.scene)
	rt.stats.lap(phasePresent)
	rt.text.EndFrame()
	rt.next()
	rt.stats.end(rt)
}

// endPass forgets the input the pass handled.
func (rt *engine) endPass() {
	rt.forgetInput()
	// Clicks the view gave its elements, which the next pass sees.
	for _, id := range rt.clickLater {
		if s := rt.states[id]; s != nil {
			s.clicks++
			s.clickMods = 0
		}
	}
	rt.clickLater = rt.clickLater[:0]
	rt.menu.chosen = 0
	// A shortcut the view did not ask for again, as it changed: what
	// this frame built may take the key (settleKeys).
	rt.unrouted = append(rt.unrouted, rt.delivered...)
	rt.delivered = rt.delivered[:0]
	rt.downs = rt.downs[:0]
	// The next pass may not ask again, as when the view cleared what asked.
	for _, e := range rt.c.reveal {
		if !slices.Contains(rt.revealIDs, e.id) {
			rt.revealIDs = append(rt.revealIDs, e.id)
		}
	}
}

// forgetInput forgets the input of the elements the pass built, which
// they handled.
func (rt *engine) forgetInput() {
	for _, s := range rt.states {
		if s.seen != rt.frame || s.pass != rt.pass {
			continue
		}
		s.clicks, s.rightClicks, s.doubleClicks = 0, 0, 0
		s.pressPending = false
		s.dragX, s.dragY = 0, 0
		s.released = false
		s.dropped = nil
		s.droppedValue, s.hasDropped = nil, false
		s.dataDropped = nil
	}
}

// prune forgets the elements the frame did not build, but those of the
// pages Routers keep.
func (rt *engine) prune() {
	if d := rt.drag; d != nil && d.native {
		if s := rt.states[d.src]; s == nil || s.seen != rt.frame || s.pass != rt.pass {
			rt.host.cancelDataDrag()
		}
	}
	unpressed := false
	for id, s := range rt.states {
		if s.seen != rt.frame || s.pass != rt.pass {
			if rt.pressed == s {
				rt.pressed, unpressed = nil, true
			}
			if !rt.keptAlive(s) {
				if s.textAdapter != nil {
					s.textAdapter.release()
				}
				delete(rt.states, id)
				if rt.scrollDrag.st == s {
					rt.scrollDrag.st = nil
				}
				// Recycle the state itself, without keeping the editor,
				// local resources or callbacks of the element that went.
				*s = state{}
				if len(rt.free) < maxFree {
					rt.free = append(rt.free, s)
				}
			}
		}
	}
	if unpressed {
		// The element pressed went away, as a button showing over a row
		// the pointer left: what the pointer is over now hovers, rather
		// than what it was over as the press began.
		var chain []uint64
		if rt.pointerIn {
			chain = rt.hitChain(rt.pointerX, rt.pointerY)
		}
		rt.setHover(chain)
	}
	rt.restoreFocus()
	// The focus does not stay in a page kept out of sight.
	if s := rt.states[rt.focused]; s != nil && (s.seen != rt.frame || s.pass != rt.pass) {
		rt.focused = 0
	}
}

// maxFree is how many pruned states the engine keeps for new elements.
const maxFree = 256

// keptAlive reports whether a state the frame did not build is in a page
// that a Router keeps: one of its history, or inside one.
func (rt *engine) keptAlive(s *state) bool {
	if len(rt.kept) == 0 {
		return false
	}
	for n := 0; s != nil && n < 64; n++ {
		if rt.kept[s.id] {
			return true
		}
		if s.page == 0 {
			return false
		}
		s = rt.states[s.page]
	}
	return false
}

// requestFrame asks the host for a frame built anew, unless one is being
// built. It comes in place of one Painter.After has due.
func (rt *engine) requestFrame() {
	if rt.inFrame {
		return
	}
	rt.redraw, rt.repaintDue = false, time.Time{}
	rt.host.requestFrame()
}

// changed asks the host for a frame built anew after the app changed what
// the view shows, from outside the view (surface.Conn.Changed).
func (rt *engine) changed() {
	rt.redraw, rt.repaintDue = false, time.Time{}
	rt.host.requestFrame()
}

// scheduleAt asks for a frame at t (After).
func (rt *engine) scheduleAt(t time.Time) {
	rt.wakeMu.Lock()
	if rt.wakeAt.IsZero() || t.Before(rt.wakeAt) {
		rt.wakeAt = t
	}
	rt.wakeMu.Unlock()
}

// armTimer arms the host's timer for the earliest frame After asked for.
func (rt *engine) armTimer() {
	rt.wakeMu.Lock()
	at := rt.wakeAt
	rt.wakeAt = time.Time{}
	rt.wakeMu.Unlock()
	rt.host.afterTimer(at)
}

// close lets the window's state go when the window closes. A window closed
// during a frame, as a close button's click closes it, keeps it until the
// frame ends: the rest of the view builds with a Context that still works,
// and reads the theme, the size and its state as before.
func (rt *engine) close() {
	if rt.inFrame {
		rt.closing = true
		return
	}
	rt.closed, rt.closing = true, false
	clear(rt.focusFields)
	rt.focusFields = nil
	for _, r := range rt.refs {
		r.closed = true
		r.requested = false
	}
	clear(rt.refs)
	rt.refs = nil
	clear(rt.parts)
	rt.parts = nil
	clear(rt.typedInputs)
	rt.typedInputs = nil
	clear(rt.notices)
	rt.notices = nil
	clear(rt.afterInputs)
	rt.afterInputs = nil
	clear(rt.inputs)
	rt.inputs = nil
	rt.arena.rt = nil
	rt.public.rt = nil
	clear(rt.actions)
	rt.actions = nil
	rt.textInputClosed = true
	if rt.drag != nil && rt.drag.native {
		rt.host.cancelDataDrag()
	}
	rt.drag, rt.incoming = nil, nil
	for _, s := range rt.states {
		if s.textAdapter != nil {
			s.textAdapter.release()
		}
	}
	rt.host.afterTimer(time.Time{})
	rt.host.repaintTimer(time.Time{})
}

// endFrame ends a frame, and closes the engine when its window closed
// during it.
func (rt *engine) endFrame() {
	rt.inFrame = false
	if rt.closing {
		rt.close()
	}
}

// commit records the laid out frame in the elements' states: their
// boxes, the hit list in paint order, the focus order.
func (rt *engine) commit(root *node, w, h float32) {
	rt.hits = rt.hits[:0]
	clear(rt.posed)
	rt.posed = rt.posed[:0]
	clear(rt.webPlaces)
	rt.webPlaces = rt.webPlaces[:0]
	rt.focusOrder, rt.focusScopes = rt.focusOrder[:0], rt.focusScopes[:0]
	rt.modal, rt.modalLayer, rt.commitScope, rt.commitPage = 0, 0, focusScope{}, 0
	if rt.groups == nil {
		rt.groups = map[uint64]groupInfo{}
	}
	clear(rt.groups)
	rt.labels = rt.labels[:0]
	// The window, which the inspector shares with the root.
	full := Rect{0, 0, w, h}
	rt.commitElement(root, full, false)
	rt.arrangeFocus()
	rt.noteGroups()
}

func (rt *engine) commitElement(e *node, clip Rect, hidden bool) {
	inline := e.isInline()
	if e.kind == kindText && e.first != nil && !inline {
		placeInline(e, e, 0)
	}
	saved, savedPage := rt.enterScope(e), rt.commitPage
	defer func() { rt.commitScope, rt.commitPage = saved, savedPage }()
	s := e.st
	s.page = rt.commitPage
	if e.flags&flagPage != 0 {
		rt.commitPage = e.id
	}
	if e.poseSet {
		rt.samplePose(e)
	}
	e.world = e.transformAt(e.x, e.y)
	if e.parent != nil && e != e.c.overlay {
		e.world = e.parent.world.Mul(e.world)
	}
	s.world = e.world
	s.clips = e.flags&(flagClipX|flagClipY|flagScrollX|flagScrollY) != 0
	if s.clips {
		s.clipRect, s.clipRadii = e.clipRect()
	}
	s.x, s.y, s.w, s.h = e.x, e.y, e.w, e.h
	if e.parent != nil {
		s.parent = e.parent.id
	} else {
		s.parent = 0
	}
	s.flags = e.flags
	s.rtl = e.rtl()
	s.anchor = 0
	if e.popover != nil {
		s.anchor = e.popover.id
	}
	if e.parent != nil && e.parent.st.flags&flagDisabled != 0 {
		// Disabled with the element around it, which may have been
		// disabled after building it, as a Fieldset.
		s.flags |= flagDisabled
	}
	s.cursor, s.tip = e.cursor, e.tip
	s.webView = e.webView
	s.role = e.role
	s.input, s.caret, s.takesText, s.noIME = e.inputFn, e.caret, e.takesText, e.noIME
	s.copyData = e.copyData
	if s.textClient != e.textClient {
		if s.textAdapter != nil {
			s.textAdapter.release()
		}
		s.textClient, s.textAdapter = e.textClient, nil
	}
	if s.textClient != nil && s.textAdapter == nil {
		s.textAdapter = &textInputAdapter{rt: rt, id: s.id}
	}
	if (e.flags&flagEditable != 0 || e.flags&flagSelectable != 0 && s.editor != nil) && s.cursor == 0 {
		s.cursor = CursorText + 1
	}
	v := intersect(e.visualRect(Rect{e.x, e.y, e.w, e.h}), clip)
	s.vx, s.vy, s.vw, s.vh = v.X, v.Y, v.W, v.H
	s.cx, s.cy = e.x+e.contentX(), e.y+e.contentY()
	s.cw, s.ch = max(e.w-e.padX(), 0), max(e.h-e.padY(), 0)
	if e.flags&(flagScrollX|flagScrollY) != 0 {
		s.contentW, s.contentH = e.contentW, e.contentH
		s.barInset = e.barInset
	}
	// What is invisible keeps its box but takes neither the pointer nor
	// the focus, and has no text to find; so does what is inert, which
	// shows.
	invisible := e.flags&(flagInvisible|flagInert) != 0 || hidden || e.backface && e.transform.facesAway()
	if invisible {
		s.vw, s.vh = 0, 0
		if rt.focused == e.id {
			rt.focused = 0
		}
	}
	if e.webView != nil {
		rt.noteWebView(e, clip, invisible)
	}
	switch {
	case e.flags&flagPassThrough != 0 || invisible:
	case inline:
		// Inline elements take the pointer over their words.
		for _, r := range e.frags {
			rt.hits = append(rt.hits, hit{s, r, e.flags})
		}
	default:
		rt.hits = append(rt.hits, hit{s, Rect{e.x, e.y, e.w, e.h}, e.flags})
	}
	label := e.label
	if f := e.nameFrom; label == "" && f != nil {
		label = f.nameOf() // an input, named by its field
	}
	if (label != "" || e.kind == kindText) && !invisible {
		if label == "" {
			label = e.text
		}
		switch {
		case !inline:
			rt.labels = append(rt.labels, labelNode{e.id, label, v})
		case len(e.frags) > 0 && (e.label != "" || e.flags&interactive != 0):
			// Their paragraph shows the text of the others.
			rt.labels = append(rt.labels, labelNode{e.id, label, intersect(e.visualRect(e.frags[0]), clip)})
		}
	}
	if e.flags&flagFocusable != 0 && !e.IsDisabled() && !invisible {
		rt.focusOrder = append(rt.focusOrder, e.id)
		rt.focusScopes = append(rt.focusScopes, rt.commitScope)
		// Tab goes to the radio button or tab chosen; not to a toggle
		// that is on.
		if g := rt.commitScope.group; g != 0 && e.checked == 2 && (e.role == RoleRadio || e.role == RoleTab) {
			if info := rt.groups[g]; info.checked == 0 {
				info.checked = e.id
				rt.groups[g] = info
			}
		}
	}
	if e.flags&(flagClipX|flagClipY|flagScrollX|flagScrollY) != 0 {
		r, _ := e.clipRect()
		clip = intersect(clip, e.visualRect(r))
	}
	// Children in flow first, absolute ones above them: the paint order.
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.flags&flagAbsolute == 0 {
			rt.commitElement(ch, clip, invisible)
		}
	}
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.flags&flagAbsolute != 0 {
			rt.commitElement(ch, clip, invisible)
		}
	}
}

func intersect(a, b Rect) Rect {
	x0, y0 := max(a.X, b.X), max(a.Y, b.Y)
	x1, y1 := min(a.X+a.W, b.X+b.W), min(a.Y+a.H, b.Y+b.H)
	if x1 <= x0 || y1 <= y0 {
		return Rect{x0, y0, 0, 0}
	}
	return Rect{x0, y0, x1 - x0, y1 - y0}
}

// updateCursor shows the cursor of the element under the pointer: the
// innermost one's that sets a cursor, up to the innermost element that
// takes clicks, whose own cursor is the default unless it sets one, so a
// button in a text field shows no I-beam.
func (rt *engine) updateCursor() {
	if rt.pressed == nil && rt.overWebView() {
		// The page sets the cursor over it: it is set again as the
		// pointer comes back.
		rt.cursor = cursorUnknown
		return
	}
	c := CursorDefault
	if rt.pressed != nil && rt.pressed.cursor != 0 {
		c = rt.pressed.cursor - 1
	} else {
		for _, id := range rt.hover {
			s := rt.states[id]
			if s == nil {
				continue
			}
			if s.cursor != 0 {
				c = s.cursor - 1
				break
			}
			if s.flags&flagClickable != 0 {
				break
			}
		}
	}
	if c != rt.cursor {
		rt.cursor = c
		rt.host.setCursor(c)
	}
}

// cursorUnknown is the cursor the engine last set while the pointer is
// over a web view, whose page set another.
const cursorUnknown Cursor = 255
