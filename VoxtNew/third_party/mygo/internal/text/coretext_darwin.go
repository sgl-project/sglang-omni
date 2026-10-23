//go:build darwin

package text

import (
	"cmp"
	"errors"
	"fmt"
	"log"
	"math"
	"runtime"
	"slices"
	"strings"
	"sync"
	"unicode"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
	"github.com/egoist/mygo/internal/scene"
	"github.com/go-text/typesetting/segmenter"
)

// Core Text lays out paragraphs: a typesetter finds the fonts, falls back
// to others for what a font lacks, shapes, and suggests line breaks, and
// Core Graphics rasterizes glyphs, in color for color fonts such as Apple
// Color Emoji, and the others with font smoothing, as AppKit draws text.
// The system font comes from NSFont, which knows its weights.

func newEngine() engine {
	e, err := newCoreText()
	if err != nil {
		log.Printf("mygo: %v: text will not show", err)
		return &stubEngine{}
	}
	return e
}

// Core Foundation, Core Text and Core Graphics structs passed by value.
type (
	cfRange           struct{ location, length int }
	cgPoint           struct{ x, y float64 }
	cgSize            struct{ w, h float64 }
	cgRect            struct{ x, y, w, h float64 }
	cgAffineTransform struct{ a, b, c, d, tx, ty float64 }

	ctParagraphStyleSetting struct {
		spec  uint32
		size  uintptr
		value unsafe.Pointer
	}
)

const (
	cfStringEncodingUTF8 = 0x08000100
	cfNumberSInt32Type   = 3
	cfNumberSInt64Type   = 4
	cfNumberFloat64Type  = 6

	ctFontUIFontSystem         = 2
	ctFontItalicTrait          = 1 << 0
	ctFontMonoSpaceTrait       = 1 << 10
	ctFontColorGlyphsTrait     = 1 << 13
	ctFontOrientationDefault   = 0
	ctRunStatusRightToLeft     = 1 << 0
	ctBaseWritingDirectionSpec = 13

	cgImageAlphaOnly             = 7
	cgImageAlphaPremultipliedLst = 1
	cgBitmapByteOrder32Big       = 4 << 12
)

var ct struct {
	// Core Foundation
	release               func(obj uintptr)
	retain                func(obj uintptr) uintptr
	hash                  func(obj uintptr) uint
	equal                 func(a, b uintptr) bool
	stringWithCharacters  func(alloc uintptr, chars *uint16, n int) uintptr
	stringWithBytes       func(alloc uintptr, bytes *byte, n int, encoding uint32, external bool) uintptr
	stringGetLength       func(s uintptr) int
	stringGetCharacters   func(s uintptr, r cfRange, out *uint16)
	dictionaryCreate      func(alloc uintptr, keys, values *uintptr, n int, keyCallbacks, valueCallbacks uintptr) uintptr
	dictionaryGetValue    func(d, key uintptr) uintptr
	arrayGetCount         func(a uintptr) int
	arrayGetValueAtIndex  func(a uintptr, i int) uintptr
	setCreate             func(alloc uintptr, values *uintptr, n int, callbacks uintptr) uintptr
	arrayCreate           func(alloc uintptr, values *uintptr, n int, callbacks uintptr) uintptr
	numberCreate          func(alloc uintptr, typ int, value unsafe.Pointer) uintptr
	numberGetValue        func(n uintptr, typ int, value unsafe.Pointer) bool
	attributedString      func(alloc, str, attrs uintptr) uintptr
	attributedMutable     func(alloc uintptr, max int, attributed uintptr) uintptr
	attributedSet         func(attributed uintptr, r cfRange, name, value uintptr)
	dataCreate            func(alloc uintptr, bytes *byte, n int) uintptr
	keyCallbacks          uintptr
	valueCallbacks        uintptr
	setCallbacks          uintptr
	arrayCallbacks        uintptr
	fontAttributeName     uintptr
	paragraphStyleName    uintptr
	familyNameAttribute   uintptr
	traitsAttribute       uintptr
	weightTrait           uintptr
	symbolicTrait         uintptr
	srgbName              uintptr
	fontWithDescriptor    func(desc uintptr, size float64, matrix uintptr) uintptr
	uiFontForLanguage     func(typ uint32, size float64, language uintptr) uintptr
	fontWithTraits        func(font uintptr, size float64, matrix uintptr, value, mask uint32) uintptr
	fontWithAttrs         func(font uintptr, size float64, matrix, desc uintptr) uintptr
	descriptorWithAttrs   func(attrs uintptr) uintptr
	descriptorMatching    func(desc, mandatory uintptr) uintptr
	descriptorAttribute   func(desc, name uintptr) uintptr
	descriptorsFromData   func(data uintptr) uintptr
	availableFamilyNames  func() uintptr
	fontGetAscent         func(font uintptr) float64
	fontGetDescent        func(font uintptr) float64
	fontGetLeading        func(font uintptr) float64
	fontGetSize           func(font uintptr) float64
	fontGetMatrix         func(font uintptr) cgAffineTransform
	fontGetSymbolicTraits func(font uintptr) uint32
	fontGetUnderlinePos   func(font uintptr) float64
	fontGetUnderlineThick func(font uintptr) float64
	fontGetXHeight        func(font uintptr) float64
	fontGetUnitsPerEm     func(font uintptr) uint32
	fontCopyTable         func(font uintptr, tag uint32, options uint32) uintptr
	fontCreatePath        func(font uintptr, glyph uint16, matrix uintptr) uintptr
	dataGetLength         func(data uintptr) int
	dataGetBytePtr        func(data uintptr) *byte
	fontGetBoundingRects  func(font uintptr, orientation uint32, glyphs *uint16, rects *cgRect, n int) cgRect
	fontDrawGlyphs        func(font uintptr, glyphs *uint16, positions *cgPoint, n int, context uintptr)
	paragraphStyleCreate  func(settings *ctParagraphStyleSetting, n int) uintptr
	typesetterCreate      func(str uintptr) uintptr
	suggestLineBreak      func(typesetter uintptr, start int, width float64) int
	typesetterCreateLine  func(typesetter uintptr, r cfRange) uintptr
	lineGetGlyphRuns      func(line uintptr) uintptr
	lineGetStringRange    func(line uintptr) cfRange
	runGetGlyphCount      func(run uintptr) int
	runGetStringRange     func(run uintptr) cfRange
	runGetStatus          func(run uintptr) uint32
	runGetAttributes      func(run uintptr) uintptr
	runGetGlyphs          func(run uintptr, r cfRange, out *uint16)
	runGetPositions       func(run uintptr, r cfRange, out *cgPoint)
	runGetAdvances        func(run uintptr, r cfRange, out *cgSize)
	runGetStringIndices   func(run uintptr, r cfRange, out *int)

	// Optional: tracking (macOS 10.12) and OpenType features by tag
	// (macOS 10.13).
	trackingName    uintptr
	kernName        uintptr
	cascadeList     uintptr
	languageName    uintptr // kCTLanguageAttributeName, macOS 10.8
	featureSettings uintptr
	featureTag      uintptr
	featureValue    uintptr

	// Core Graphics
	colorSpaceWithName      func(name uintptr) uintptr
	colorSpaceDeviceRGB     func() uintptr
	bitmapContextCreate     func(data unsafe.Pointer, w, h, bitsPerComponent, bytesPerRow int, space uintptr, info uint32) uintptr
	contextRelease          func(ctx uintptr)
	contextAntialias        func(ctx uintptr, on bool)
	contextSmoothFonts      func(ctx uintptr, on bool)
	contextAllowSubpixelPos func(ctx uintptr, on bool)
	contextSubpixelPos      func(ctx uintptr, on bool)
	contextAllowQuantize    func(ctx uintptr, on bool)
	contextQuantize         func(ctx uintptr, on bool)
	contextScaleCTM         func(ctx uintptr, sx, sy float64)
	contextSetFill          func(ctx uintptr, r, g, b, a float64)
	pathApply               func(path, info, fn uintptr)
	pathRelease             func(path uintptr)

	// Objective-C
	poolPush func() uintptr
	poolPop  func(pool uintptr)
}

func loadCoreText() error {
	var missing []string
	lib := func(path string) uintptr {
		h, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			missing = append(missing, path)
		}
		return h
	}
	bind := func(lib uintptr, fn any, name string) {
		sym, err := purego.Dlsym(lib, name)
		if err != nil || sym == 0 {
			missing = append(missing, name)
			return
		}
		purego.RegisterFunc(fn, sym)
	}
	// Constants are pointers to CFStringRefs, or callback structs.
	addr := func(lib uintptr, name string) uintptr {
		sym, err := purego.Dlsym(lib, name)
		if err != nil || sym == 0 {
			missing = append(missing, name)
		}
		return sym
	}
	value := func(lib uintptr, name string) uintptr {
		p := addr(lib, name)
		if p == 0 {
			return 0
		}
		return **(**uintptr)(unsafe.Pointer(&p))
	}
	cf := lib("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
	text := lib("/System/Library/Frameworks/CoreText.framework/CoreText")
	cg := lib("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
	objcLib := lib("/usr/lib/libobjc.A.dylib")
	lib("/System/Library/Frameworks/AppKit.framework/AppKit") // NSFont
	if len(missing) > 0 {
		return fmt.Errorf("cannot load %s", strings.Join(missing, ", "))
	}
	bindDirect(cf, text, addr)
	bind(cf, &ct.dataGetLength, "CFDataGetLength")
	bind(cf, &ct.dataGetBytePtr, "CFDataGetBytePtr")
	bind(text, &ct.fontGetUnderlinePos, "CTFontGetUnderlinePosition")
	bind(text, &ct.fontGetUnderlineThick, "CTFontGetUnderlineThickness")
	bind(text, &ct.fontGetXHeight, "CTFontGetXHeight")
	bind(text, &ct.fontGetUnitsPerEm, "CTFontGetUnitsPerEm")
	bind(text, &ct.fontCopyTable, "CTFontCopyTable")
	bind(text, &ct.fontCreatePath, "CTFontCreatePathForGlyph")
	bind(cg, &ct.pathApply, "CGPathApply")
	bind(cg, &ct.pathRelease, "CGPathRelease")
	bind(cf, &ct.stringWithBytes, "CFStringCreateWithBytes")
	bind(cf, &ct.stringGetLength, "CFStringGetLength")
	bind(cf, &ct.stringGetCharacters, "CFStringGetCharacters")
	bind(cf, &ct.setCreate, "CFSetCreate")
	bind(cf, &ct.arrayCreate, "CFArrayCreate")
	bind(cf, &ct.numberGetValue, "CFNumberGetValue")
	bind(cf, &ct.dataCreate, "CFDataCreate")
	ct.keyCallbacks = addr(cf, "kCFTypeDictionaryKeyCallBacks")
	ct.valueCallbacks = addr(cf, "kCFTypeDictionaryValueCallBacks")
	ct.setCallbacks = addr(cf, "kCFTypeSetCallBacks")
	ct.arrayCallbacks = addr(cf, "kCFTypeArrayCallBacks")
	ct.fontAttributeName = value(text, "kCTFontAttributeName")
	ct.paragraphStyleName = value(text, "kCTParagraphStyleAttributeName")
	ct.familyNameAttribute = value(text, "kCTFontFamilyNameAttribute")
	ct.traitsAttribute = value(text, "kCTFontTraitsAttribute")
	ct.weightTrait = value(text, "kCTFontWeightTrait")
	ct.symbolicTrait = value(text, "kCTFontSymbolicTrait")
	bind(text, &ct.fontWithDescriptor, "CTFontCreateWithFontDescriptor")
	bind(text, &ct.availableFamilyNames, "CTFontManagerCopyAvailableFontFamilyNames")
	bind(text, &ct.uiFontForLanguage, "CTFontCreateUIFontForLanguage")
	bind(text, &ct.fontWithTraits, "CTFontCreateCopyWithSymbolicTraits")
	bind(text, &ct.fontWithAttrs, "CTFontCreateCopyWithAttributes")
	bind(text, &ct.descriptorWithAttrs, "CTFontDescriptorCreateWithAttributes")
	bind(text, &ct.descriptorMatching, "CTFontDescriptorCreateMatchingFontDescriptor")
	bind(text, &ct.descriptorAttribute, "CTFontDescriptorCopyAttribute")
	bind(text, &ct.fontGetAscent, "CTFontGetAscent")
	bind(text, &ct.fontGetDescent, "CTFontGetDescent")
	bind(text, &ct.fontGetLeading, "CTFontGetLeading")
	bind(text, &ct.fontGetSize, "CTFontGetSize")
	bind(text, &ct.fontGetMatrix, "CTFontGetMatrix")
	bind(text, &ct.fontGetSymbolicTraits, "CTFontGetSymbolicTraits")
	bind(text, &ct.fontGetBoundingRects, "CTFontGetBoundingRectsForGlyphs")
	bind(text, &ct.fontDrawGlyphs, "CTFontDrawGlyphs")
	bind(text, &ct.paragraphStyleCreate, "CTParagraphStyleCreate")
	bind(text, &ct.suggestLineBreak, "CTTypesetterSuggestLineBreak")
	bind(cg, &ct.colorSpaceDeviceRGB, "CGColorSpaceCreateDeviceRGB")
	bind(cg, &ct.bitmapContextCreate, "CGBitmapContextCreate")
	bind(cg, &ct.contextRelease, "CGContextRelease")
	bind(cg, &ct.contextAntialias, "CGContextSetShouldAntialias")
	bind(cg, &ct.contextSmoothFonts, "CGContextSetShouldSmoothFonts")
	bind(cg, &ct.contextAllowSubpixelPos, "CGContextSetAllowsFontSubpixelPositioning")
	bind(cg, &ct.contextSubpixelPos, "CGContextSetShouldSubpixelPositionFonts")
	bind(cg, &ct.contextAllowQuantize, "CGContextSetAllowsFontSubpixelQuantization")
	bind(cg, &ct.contextQuantize, "CGContextSetShouldSubpixelQuantizeFonts")
	bind(cg, &ct.contextScaleCTM, "CGContextScaleCTM")
	bind(cg, &ct.contextSetFill, "CGContextSetRGBFillColor")
	bind(objcLib, &ct.poolPush, "objc_autoreleasePoolPush")
	bind(objcLib, &ct.poolPop, "objc_autoreleasePoolPop")
	if len(missing) > 0 {
		return fmt.Errorf("missing %s", strings.Join(missing, ", "))
	}
	optional := func(lib uintptr, name string) uintptr {
		if p, err := purego.Dlsym(lib, name); err == nil && p != 0 {
			return **(**uintptr)(unsafe.Pointer(&p))
		}
		return 0
	}
	ct.trackingName = optional(text, "kCTTrackingAttributeName")
	ct.kernName = optional(text, "kCTKernAttributeName")
	ct.cascadeList = optional(text, "kCTFontCascadeListAttribute")
	ct.languageName = optional(text, "kCTLanguageAttributeName")
	ct.featureSettings = optional(text, "kCTFontFeatureSettingsAttribute")
	ct.featureTag = optional(text, "kCTFontOpenTypeFeatureTag")
	ct.featureValue = optional(text, "kCTFontOpenTypeFeatureValue")
	initDelegates(text, optional)
	// Optional: adding fonts from memory (macOS 10.13), sRGB by name.
	if sym, err := purego.Dlsym(text, "CTFontManagerCreateFontDescriptorsFromData"); err == nil && sym != 0 {
		purego.RegisterFunc(&ct.descriptorsFromData, sym)
	}
	if sym, err := purego.Dlsym(cg, "CGColorSpaceCreateWithName"); err == nil && sym != 0 {
		purego.RegisterFunc(&ct.colorSpaceWithName, sym)
		if p, err := purego.Dlsym(cg, "kCGColorSpaceSRGB"); err == nil && p != 0 {
			ct.srgbName = **(**uintptr)(unsafe.Pointer(&p))
		}
	}
	return nil
}

// bindDirect binds the functions of Core Foundation and Core Text that
// each layout calls to call them through purego.SyscallN, which allocates
// once a call where RegisterFunc's reflection allocates three or more
// times. They take integers, pointers and CFRanges, which C passes in two
// integer registers and returns in the first two on arm64 and amd64, as
// SyscallN's r1 and r2.
func bindDirect(cf, text uintptr, addr func(lib uintptr, name string) uintptr) {
	release, retain := addr(cf, "CFRelease"), addr(cf, "CFRetain")
	ct.release = func(obj uintptr) { purego.SyscallN(release, obj) }
	ct.retain = func(obj uintptr) uintptr { r, _, _ := purego.SyscallN(retain, obj); return r }
	hash, equal := addr(cf, "CFHash"), addr(cf, "CFEqual")
	ct.hash = func(obj uintptr) uint { r, _, _ := purego.SyscallN(hash, obj); return uint(r) }
	ct.equal = func(a, b uintptr) bool { r, _, _ := purego.SyscallN(equal, a, b); return uint8(r) != 0 }
	stringWithCharacters := addr(cf, "CFStringCreateWithCharacters")
	ct.stringWithCharacters = func(alloc uintptr, chars *uint16, n int) uintptr {
		r, _, _ := purego.SyscallN(stringWithCharacters, alloc, uintptr(unsafe.Pointer(chars)), uintptr(n))
		return r
	}
	dictionaryCreate, dictionaryGetValue := addr(cf, "CFDictionaryCreate"), addr(cf, "CFDictionaryGetValue")
	ct.dictionaryCreate = func(alloc uintptr, keys, values *uintptr, n int, keyCallbacks, valueCallbacks uintptr) uintptr {
		r, _, _ := purego.SyscallN(dictionaryCreate, alloc, uintptr(unsafe.Pointer(keys)), uintptr(unsafe.Pointer(values)), uintptr(n), keyCallbacks, valueCallbacks)
		return r
	}
	ct.dictionaryGetValue = func(d, key uintptr) uintptr { r, _, _ := purego.SyscallN(dictionaryGetValue, d, key); return r }
	arrayGetCount, arrayGetValueAtIndex := addr(cf, "CFArrayGetCount"), addr(cf, "CFArrayGetValueAtIndex")
	ct.arrayGetCount = func(a uintptr) int { r, _, _ := purego.SyscallN(arrayGetCount, a); return int(r) }
	ct.arrayGetValueAtIndex = func(a uintptr, i int) uintptr {
		r, _, _ := purego.SyscallN(arrayGetValueAtIndex, a, uintptr(i))
		return r
	}
	numberCreate := addr(cf, "CFNumberCreate")
	ct.numberCreate = func(alloc uintptr, typ int, value unsafe.Pointer) uintptr {
		r, _, _ := purego.SyscallN(numberCreate, alloc, uintptr(typ), uintptr(value))
		return r
	}
	attributedString, attributedMutable := addr(cf, "CFAttributedStringCreate"), addr(cf, "CFAttributedStringCreateMutableCopy")
	attributedSet := addr(cf, "CFAttributedStringSetAttribute")
	ct.attributedString = func(alloc, str, attrs uintptr) uintptr {
		r, _, _ := purego.SyscallN(attributedString, alloc, str, attrs)
		return r
	}
	ct.attributedMutable = func(alloc uintptr, max int, attributed uintptr) uintptr {
		r, _, _ := purego.SyscallN(attributedMutable, alloc, uintptr(max), attributed)
		return r
	}
	ct.attributedSet = func(attributed uintptr, r cfRange, name, value uintptr) {
		purego.SyscallN(attributedSet, attributed, uintptr(r.location), uintptr(r.length), name, value)
	}
	typesetterCreate, typesetterCreateLine := addr(text, "CTTypesetterCreateWithAttributedString"), addr(text, "CTTypesetterCreateLine")
	ct.typesetterCreate = func(str uintptr) uintptr { r, _, _ := purego.SyscallN(typesetterCreate, str); return r }
	ct.typesetterCreateLine = func(typesetter uintptr, r cfRange) uintptr {
		line, _, _ := purego.SyscallN(typesetterCreateLine, typesetter, uintptr(r.location), uintptr(r.length))
		return line
	}
	lineGetGlyphRuns, lineGetStringRange := addr(text, "CTLineGetGlyphRuns"), addr(text, "CTLineGetStringRange")
	ct.lineGetGlyphRuns = func(line uintptr) uintptr { r, _, _ := purego.SyscallN(lineGetGlyphRuns, line); return r }
	ct.lineGetStringRange = func(line uintptr) cfRange {
		loc, n, _ := purego.SyscallN(lineGetStringRange, line)
		return cfRange{int(loc), int(n)}
	}
	runGetGlyphCount, runGetStringRange := addr(text, "CTRunGetGlyphCount"), addr(text, "CTRunGetStringRange")
	runGetStatus, runGetAttributes := addr(text, "CTRunGetStatus"), addr(text, "CTRunGetAttributes")
	ct.runGetGlyphCount = func(run uintptr) int { r, _, _ := purego.SyscallN(runGetGlyphCount, run); return int(r) }
	ct.runGetStringRange = func(run uintptr) cfRange {
		loc, n, _ := purego.SyscallN(runGetStringRange, run)
		return cfRange{int(loc), int(n)}
	}
	ct.runGetStatus = func(run uintptr) uint32 { r, _, _ := purego.SyscallN(runGetStatus, run); return uint32(r) }
	ct.runGetAttributes = func(run uintptr) uintptr { r, _, _ := purego.SyscallN(runGetAttributes, run); return r }
	runGetGlyphs, runGetPositions := addr(text, "CTRunGetGlyphs"), addr(text, "CTRunGetPositions")
	runGetAdvances, runGetStringIndices := addr(text, "CTRunGetAdvances"), addr(text, "CTRunGetStringIndices")
	ct.runGetGlyphs = func(run uintptr, r cfRange, out *uint16) {
		purego.SyscallN(runGetGlyphs, run, uintptr(r.location), uintptr(r.length), uintptr(unsafe.Pointer(out)))
	}
	ct.runGetPositions = func(run uintptr, r cfRange, out *cgPoint) {
		purego.SyscallN(runGetPositions, run, uintptr(r.location), uintptr(r.length), uintptr(unsafe.Pointer(out)))
	}
	ct.runGetAdvances = func(run uintptr, r cfRange, out *cgSize) {
		purego.SyscallN(runGetAdvances, run, uintptr(r.location), uintptr(r.length), uintptr(unsafe.Pointer(out)))
	}
	ct.runGetStringIndices = func(run uintptr, r cfRange, out *int) {
		purego.SyscallN(runGetStringIndices, run, uintptr(r.location), uintptr(r.length), uintptr(unsafe.Pointer(out)))
	}
}

type coreText struct {
	shapeScratch
	styles [2]uintptr // paragraph styles: left-to-right, right-to-left
	srgb   uintptr
	smooth bool // the user leaves font smoothing on
	// bands caches where glyphs' outlines cross underlines (inkInBand).
	bands map[bandKey][2]float32

	primary map[Style]uintptr // the CTFont of each style
	fonts   map[uint][]*Font  // by CFHash of their CTFont
	// native finds Fonts by their CTFont, which they retain.
	native map[uintptr]*Font
	// oblique maps an upright Font to its synthetic italic variant. Both
	// are retained by fonts, and have distinct glyph/outline cache keys.
	oblique map[*Font]*Font
	// italic names an attribute that marks the runes of italic spans in a
	// text that is not italic, owned, so that Core Text ends a run where
	// italics start or stop even when both sides are in one font, as a
	// CJK font without an italic face. Any value marks them; it is its
	// own.
	italic uintptr
	// attrs are the attributes of strings in a font of primary, in each
	// direction, without tracking or kerning, owned.
	attrs map[attrsKey]uintptr
	// seg and breaks find where lines may break (lineBreaks).
	seg    segmenter.Segmenter
	breaks []bool
	color  map[uintptr]bool

	// Buffers shape reuses: the text in UTF-16 with the rune of each code
	// unit, the attributes of the string, the code unit of each rune, and
	// what Core Text tells of a run.
	u16          []uint16
	index, at    []int
	keys, values []uintptr
	runIDs       []uint16
	runPoints    []cgPoint
	runAdvances  []cgSize
	runIndices   []int

	registered map[string][]registeredFace // by lowercased family
	// names are the families the app's fonts are listed under, one per
	// face: the name it gave them, or their own.
	names []string
}

type attrsKey struct {
	font uintptr
	rtl  bool
	lang string
}

type registeredFace struct {
	desc   uintptr
	weight float64 // -1 to 1
	italic bool
	mono   bool
}

func newCoreText() (*coreText, error) {
	if err := loadCoreText(); err != nil {
		return nil, err
	}
	e := &coreText{
		primary:    map[Style]uintptr{},
		fonts:      map[uint][]*Font{},
		native:     map[uintptr]*Font{},
		oblique:    map[*Font]*Font{},
		attrs:      map[attrsKey]uintptr{},
		color:      map[uintptr]bool{},
		registered: map[string][]registeredFace{},
		bands:      map[bandKey][2]float32{},
		italic:     cfString("MyGoItalic"),
	}
	for i, dir := range [2]int8{0, 1} { // kCTWritingDirectionLeftToRight, RightToLeft
		d := dir
		s := ctParagraphStyleSetting{spec: ctBaseWritingDirectionSpec, size: 1, value: unsafe.Pointer(&d)}
		e.styles[i] = ct.paragraphStyleCreate(&s, 1)
		runtime.KeepAlive(&d)
	}
	if ct.colorSpaceWithName != nil && ct.srgbName != 0 {
		e.srgb = ct.colorSpaceWithName(ct.srgbName)
	}
	if e.srgb == 0 {
		e.srgb = ct.colorSpaceDeviceRGB()
	}
	e.smooth = fontSmoothing()
	return e, nil
}

// fontSmoothing reports whether the user leaves font smoothing on: AppKit
// turns it off when AppleFontSmoothing is 0, or not a number, which Core
// Graphics' bitmap contexts do not heed.
func fontSmoothing() bool {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := ct.poolPush()
	defer ct.poolPop(pool)
	cls := objc.ID(objc.GetClass("NSUserDefaults"))
	if cls == 0 {
		return true
	}
	defaults := cls.Send(objc.RegisterName("standardUserDefaults"))
	key := cfString("AppleFontSmoothing")
	defer ct.release(key)
	if defaults.Send(objc.RegisterName("objectForKey:"), key) == 0 {
		return true
	}
	return objc.Send[int](defaults, objc.RegisterName("integerForKey:"), key) != 0
}

func cfString(s string) uintptr {
	if s == "" {
		return ct.stringWithBytes(0, nil, 0, cfStringEncodingUTF8, false)
	}
	b := []byte(s)
	return ct.stringWithBytes(0, &b[0], len(b), cfStringEncodingUTF8, false)
}

func goString(s uintptr) string {
	n := ct.stringGetLength(s)
	if n <= 0 {
		return ""
	}
	buf := make([]uint16, n)
	ct.stringGetCharacters(s, cfRange{0, n}, &buf[0])
	return string(utf16Decode(buf))
}

func utf16Decode(u []uint16) []rune {
	out := make([]rune, 0, len(u))
	for i := 0; i < len(u); i++ {
		r := rune(u[i])
		if r >= 0xD800 && r < 0xDC00 && i+1 < len(u) && u[i+1] >= 0xDC00 && u[i+1] < 0xE000 {
			r = (r-0xD800)<<10 + rune(u[i+1]) - 0xDC00 + 0x10000
			i++
		}
		out = append(out, r)
	}
	return out
}

func cfDictionary(keys, values []uintptr) uintptr {
	return ct.dictionaryCreate(0, &keys[0], &values[0], len(keys), ct.keyCallbacks, ct.valueCallbacks)
}

func cfFloat(v float64) uintptr { return ct.numberCreate(0, cfNumberFloat64Type, unsafe.Pointer(&v)) }

// nsWeight converts a CSS weight to NSFont's (and Core Text's) scale, from
// -1 to 1.
func nsWeight(w int) float64 {
	steps := [...]float64{-0.8, -0.6, -0.4, 0, 0.23, 0.3, 0.4, 0.56, 0.62}
	f := (float64(max(100, min(w, 900))) - 100) / 100
	i := min(int(f), len(steps)-2)
	return steps[i] + (steps[i+1]-steps[i])*(f-float64(i))
}

// systemFont returns the system font, or its monospaced kind, owned.
func (e *coreText) systemFont(size float32, weight int, italic, mono bool) uintptr {
	// An autorelease pool belongs to a thread, which the goroutine must
	// not leave before popping it, or Objective-C crashes.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := ct.poolPush()
	defer ct.poolPop(pool)
	sel := "systemFontOfSize:weight:"
	if mono {
		sel = "monospacedSystemFontOfSize:weight:"
	}
	var font uintptr
	if cls := objc.ID(objc.GetClass("NSFont")); cls != 0 && objc.Send[bool](cls, objc.RegisterName("respondsToSelector:"), objc.RegisterName(sel)) {
		if font = uintptr(cls.Send(objc.RegisterName(sel), float64(size), nsWeight(weight))); font != 0 {
			ct.retain(font)
		}
	}
	if font == 0 {
		if mono {
			font = e.named("Menlo", size, weight, false)
		} else {
			font = ct.uiFontForLanguage(ctFontUIFontSystem, float64(size), 0)
		}
	}
	if font != 0 && italic {
		if f := ct.fontWithTraits(font, 0, 0, ctFontItalicTrait, ctFontItalicTrait); f != 0 {
			ct.release(font)
			font = f
		}
	}
	return font
}

// named returns a font of a family the system has, owned, or 0.
func (e *coreText) named(family string, size float32, weight int, italic bool) uintptr {
	desc := namedDesc(family, weight, italic)
	if desc == 0 {
		return 0
	}
	defer ct.release(desc)
	return ct.fontWithDescriptor(desc, float64(size), 0)
}

// namedDesc returns the descriptor of the face of a family the system has
// that best matches a weight and italics, owned, or 0.
func namedDesc(family string, weight int, italic bool) uintptr {
	name := cfString(family)
	defer ct.release(name)
	symbolic := int32(0)
	if italic {
		symbolic = ctFontItalicTrait
	}
	w := cfFloat(nsWeight(weight))
	defer ct.release(w)
	s := ct.numberCreate(0, cfNumberSInt32Type, unsafe.Pointer(&symbolic))
	defer ct.release(s)
	traits := cfDictionary([]uintptr{ct.weightTrait, ct.symbolicTrait}, []uintptr{w, s})
	defer ct.release(traits)
	attrs := cfDictionary([]uintptr{ct.familyNameAttribute, ct.traitsAttribute}, []uintptr{name, traits})
	defer ct.release(attrs)
	desc := ct.descriptorWithAttrs(attrs)
	if desc == 0 {
		return 0
	}
	defer ct.release(desc)
	mandatory := []uintptr{ct.familyNameAttribute}
	set := ct.setCreate(0, &mandatory[0], 1, ct.setCallbacks)
	defer ct.release(set)
	return ct.descriptorMatching(desc, set)
}

// ctFont returns the CTFont of a style.
func (e *coreText) ctFont(style Style) uintptr {
	key := Style{Family: style.Family, Size: style.FontSize(), Weight: style.weight(), Italic: style.Italic, Features: style.Features}
	if f, ok := e.primary[key]; ok {
		return f
	}
	if len(e.primary) >= 256 {
		e.forgetPrimary()
	}
	if key.Features != "" {
		// The font without the features, with them.
		base := key
		base.Features = ""
		font := e.ctFont(base)
		if with := withFeatures(font, features(key.Features)); with != 0 {
			font = with
		} else {
			ct.retain(font)
		}
		e.primary[key] = font
		return font
	}
	var font uintptr
	families, chosen := familyList(key.Family), -1
	for i, family := range families {
		chosen = i
		switch generic(family) {
		case "system-ui", "sans-serif":
			font = e.systemFont(key.Size, key.Weight, key.Italic, false)
		case "monospace":
			font = e.systemFont(key.Size, key.Weight, key.Italic, true)
		case "serif":
			for _, name := range []string{"Times New Roman", "Times", "Georgia"} {
				if font = e.named(name, key.Size, key.Weight, key.Italic); font != 0 {
					break
				}
			}
		default:
			if faces, ok := e.registered[strings.ToLower(family)]; ok {
				font = registeredFont(faces, key)
			} else {
				font = e.named(family, key.Size, key.Weight, key.Italic)
			}
		}
		if font != 0 {
			break
		}
	}
	if font == 0 {
		font = e.systemFont(key.Size, key.Weight, key.Italic, false)
	} else if ct.cascadeList != 0 {
		// The families after it come before the system's for what it
		// lacks.
		var descs []uintptr
		for _, family := range families[chosen+1:] {
			if generic(family) != "" {
				continue
			}
			if faces, ok := e.registered[strings.ToLower(family)]; ok {
				d := registeredDesc(faces, key)
				ct.retain(d)
				descs = append(descs, d)
			} else if d := namedDesc(family, key.Weight, key.Italic); d != 0 {
				descs = append(descs, d)
			}
		}
		if len(descs) > 0 {
			array := ct.arrayCreate(0, &descs[0], len(descs), ct.arrayCallbacks)
			attrs := cfDictionary([]uintptr{ct.cascadeList}, []uintptr{array})
			desc := ct.descriptorWithAttrs(attrs)
			if with := ct.fontWithAttrs(font, 0, 0, desc); with != 0 {
				ct.release(font)
				font = with
			}
			for _, obj := range append(descs, array, attrs, desc) {
				ct.release(obj)
			}
		}
	}
	e.primary[key] = font
	return font
}

// obliqueSkew slants synthetic italics by 14°, as WebKit does.
var obliqueSkew = math.Tan(14 * math.Pi / 180)

// syntheticItalic returns a slanted copy of a font without an italic face,
// owned, or 0 when it has one. Core Text's y axis points up, so a positive
// c moves the glyph's top right. A font already transformed is left as it
// is, and color glyphs stay upright, as Core Text draws their bitmaps.
func syntheticItalic(font uintptr) uintptr {
	if font == 0 || ct.fontGetSymbolicTraits(font)&(ctFontItalicTrait|ctFontColorGlyphsTrait) != 0 {
		return 0
	}
	matrix := ct.fontGetMatrix(font)
	if matrix.c != 0 {
		return 0
	}
	matrix.c = obliqueSkew
	with := ct.fontWithAttrs(font, 0, uintptr(unsafe.Pointer(&matrix)), 0)
	runtime.KeepAlive(&matrix)
	return with
}

// runFont returns the Font of a run's CTFont, slanted when the run is
// italic and the font has no italic face: the font of an italic style, or
// one Core Text fell back to for glyphs it lacks, as CJK fonts.
func (e *coreText) runFont(font uintptr, italic bool) *Font {
	base := e.fontOf(font)
	if !italic {
		return base
	}
	if f, ok := e.oblique[base]; ok {
		return f
	}
	f := base
	if with := syntheticItalic(font); with != 0 {
		f = e.fontOf(with)
		ct.release(with)
	}
	e.oblique[base] = f
	return f
}

// styleSpans returns a copy of the attributed string of text, of style,
// with the font and tracking of each span over its runes, owned.
func (e *coreText) styleSpans(attributed uintptr, style Style, spans []Span, text []rune) uintptr {
	styled := ct.attributedMutable(0, 0, attributed)
	// The UTF-16 code unit of each rune.
	e.at = slices.Grow(e.at[:0], len(text)+1)[:len(text)+1]
	at := e.at
	at[0] = 0
	for i, r := range text {
		at[i+1] = at[i] + 1
		if r >= 0x10000 {
			at[i+1]++
		}
	}
	objects := 0
	for i, sp := range spans {
		if sp.ObjectWidth > 0 {
			start := 0
			if i > 0 {
				start = spans[i-1].End
			}
			objects += max(min(sp.End, len(text))-start, 0)
		}
	}
	runDelegates.reset(objects)
	objects = 0
	from := 0
	for _, sp := range spans {
		to := max(from, min(sp.End, len(text)))
		r := cfRange{at[from], at[to] - at[from]}
		start := from
		from = to
		if r.length == 0 {
			continue
		}
		if sp.ObjectWidth > 0 {
			// Each rune is an object of its own: a run Core Text sizes by
			// its delegate.
			for k := start; k < to; k++ {
				if d := runDelegates.delegate(objects, sp.ObjectWidth, sp.ObjectHeight); d != 0 {
					ct.attributedSet(styled, cfRange{at[k], at[k+1] - at[k]}, runDelegates.attribute, d)
					ct.release(d)
				}
				objects++
			}
		}
		s := sp.style(style)
		if font := e.ctFont(s); font != 0 {
			ct.attributedSet(styled, r, ct.fontAttributeName, font)
		}
		if s.Italic && !style.Italic {
			ct.attributedSet(styled, r, e.italic, e.italic)
		}
		if s.LetterSpacing != style.LetterSpacing && ct.trackingName != 0 {
			tracking := cfFloat(float64(s.LetterSpacing))
			ct.attributedSet(styled, r, ct.trackingName, tracking)
			ct.release(tracking)
		}
		if noKerning(sp.Features) && ct.kernName != 0 {
			zero := cfFloat(0)
			ct.attributedSet(styled, r, ct.kernName, zero)
			ct.release(zero)
		}
		if s.Lang != style.Lang && s.Lang != "" && ct.languageName != 0 {
			tag := cfString(s.Lang)
			ct.attributedSet(styled, r, ct.languageName, tag)
			ct.release(tag)
		}
	}
	return styled
}

// noKerning reports whether features turn kerning off, which Core Text
// does by a kern of 0 rather than by the feature.
func noKerning(list string) bool {
	for _, f := range features(list) {
		if f.tag == [4]byte{'k', 'e', 'r', 'n'} && f.value == 0 {
			return true
		}
	}
	return false
}

// withFeatures returns a copy of font with OpenType features, owned, or 0
// when Core Text takes no features by tag.
func withFeatures(font uintptr, fs []feature) uintptr {
	if len(fs) == 0 || ct.featureSettings == 0 || ct.featureTag == 0 || ct.featureValue == 0 {
		return 0
	}
	settings := make([]uintptr, len(fs))
	for i, f := range fs {
		tag := cfString(string(f.tag[:]))
		v := int64(f.value)
		value := ct.numberCreate(0, cfNumberSInt64Type, unsafe.Pointer(&v))
		settings[i] = cfDictionary([]uintptr{ct.featureTag, ct.featureValue}, []uintptr{tag, value})
		ct.release(tag)
		ct.release(value)
	}
	array := ct.arrayCreate(0, &settings[0], len(settings), ct.arrayCallbacks)
	for _, s := range settings {
		ct.release(s)
	}
	attrs := cfDictionary([]uintptr{ct.featureSettings}, []uintptr{array})
	ct.release(array)
	desc := ct.descriptorWithAttrs(attrs)
	ct.release(attrs)
	defer ct.release(desc)
	return ct.fontWithAttrs(font, 0, 0, desc) // size 0 keeps the font's
}

// registeredFont returns the font of the face of a registered family that
// best matches a style, owned.
func registeredFont(faces []registeredFace, style Style) uintptr {
	return ct.fontWithDescriptor(registeredDesc(faces, style), float64(style.FontSize()), 0)
}

// registeredDesc returns the descriptor of the face of a registered family
// that best matches a style, not owned.
func registeredDesc(faces []registeredFace, style Style) uintptr {
	want := nsWeight(style.weight())
	best := -1
	score := func(f registeredFace) float64 {
		s := math.Abs(f.weight - want)
		if f.italic != style.Italic {
			s += 4
		}
		return s
	}
	for i, f := range faces {
		if best < 0 || score(f) < score(faces[best]) {
			best = i
		}
	}
	return faces[best].desc
}

func (e *coreText) font(style Style) *Font {
	f := e.ctFont(style)
	if f == 0 {
		return nil
	}
	return e.fontOf(f)
}

// fontOf returns the Font of a CTFont.
func (e *coreText) fontOf(font uintptr) *Font {
	if f, ok := e.native[font]; ok {
		return f
	}
	h := ct.hash(font)
	for _, f := range e.fonts[h] {
		if f.native == font || ct.equal(f.native, font) {
			return f
		}
	}
	ct.retain(font)
	f := &Font{
		Size:    float32(ct.fontGetSize(font)),
		Ascent:  float32(ct.fontGetAscent(font)),
		Descent: float32(ct.fontGetDescent(font)),
		LineGap: float32(ct.fontGetLeading(font)),
		native:  font,
		shaded:  e.smooth && !e.isColor(font),
		// Smoothing thickens glyphs, whatever the user's setting.
		thickens: !e.isColor(font),
	}
	e.fonts[h] = append(e.fonts[h], f)
	e.native[font] = f
	return f
}

func (e *coreText) shape(text []rune, style Style, spans []Span, width float32, rtl, wholeWords bool) []shapedLine {
	font := e.ctFont(style)
	if font == 0 || len(text) == 0 {
		return nil
	}
	e.u16, e.index = appendUTF16(e.u16[:0], e.index[:0], text)
	u16, index := e.u16, e.index
	n := len(u16)
	str := ct.stringWithCharacters(0, &u16[0], n)
	defer ct.release(str)
	paragraph := e.styles[0]
	if rtl {
		paragraph = e.styles[1]
	}
	lang := style.Lang
	if ct.languageName == 0 {
		lang = ""
	}
	key := attrsKey{font, rtl, lang}
	attrs, cached := e.attrs[key]
	keys, values := append(e.keys[:0], ct.fontAttributeName, ct.paragraphStyleName), append(e.values[:0], font, paragraph)
	base := 2
	if lang != "" {
		// The language picks the glyphs of a font that has several, as
		// Han characters in Chinese and Japanese, and the fonts Core Text
		// falls back to.
		tag := cfString(lang)
		defer ct.release(tag)
		keys, values = append(keys, ct.languageName), append(values, tag)
		base++
	}
	if style.LetterSpacing != 0 && ct.trackingName != 0 {
		// Tracking, in points as DIPs, keeps the font's kerning.
		tracking := cfFloat(float64(style.LetterSpacing))
		defer ct.release(tracking)
		keys, values = append(keys, ct.trackingName), append(values, tracking)
	}
	if noKerning(style.Features) && ct.kernName != 0 {
		zero := cfFloat(0)
		defer ct.release(zero)
		keys, values = append(keys, ct.kernName), append(values, zero)
	}
	e.keys, e.values = keys, values
	switch {
	case len(keys) > base:
		attrs = cfDictionary(keys, values)
		defer ct.release(attrs)
	case !cached:
		// The font of a style lives in primary, with its attributes.
		attrs = cfDictionary(keys, values)
		e.attrs[key] = attrs
	}
	attributed := ct.attributedString(0, str, attrs)
	defer ct.release(attributed)
	if len(spans) > 0 {
		styled := e.styleSpans(attributed, style, spans, text)
		defer ct.release(styled)
		attributed = styled
	}
	typesetter := ct.typesetterCreate(attributed)
	if typesetter == 0 {
		return nil
	}
	defer ct.release(typesetter)
	var breaks []bool
	if wholeWords && width > 0 {
		e.breaks = lineBreaks(&e.seg, e.breaks, text)
		breaks = e.breaks
	}
	// Widths are float32, from Core Text's doubles rounded glyph by glyph:
	// a text laid out again at the width it measured may come out short
	// of Core Text's own by more than the 0.0002 DIPs it forgives.
	breakWidth := float64(width) * (1 + 1e-6)
	var lines []shapedLine
	mark := len(e.shapeScratch.lines)
	for start := 0; start < n; {
		count := n - start
		if width > 0 {
			count = max(ct.suggestLineBreak(typesetter, start, breakWidth), 1)
			// Core Text breaks a word that does not fit a line; carry
			// on to the next break instead.
			if breaks != nil {
				end := min(start+count, n)
				for end < n && (index[end] == index[end-1] || !breaks[index[end]]) {
					end++
				}
				count = end - start
			}
		}
		line := ct.typesetterCreateLine(typesetter, cfRange{start, count})
		if line == 0 {
			break
		}
		lines = e.addLine(mark, e.line(line, index, n, style.Italic))
		ct.release(line)
		start += count
	}
	return lines
}

// line reads the runs of a line of a text that italic says is italic, or
// whose runs styleSpans marked as such.
func (e *coreText) line(line uintptr, index []int, n int, italic bool) shapedLine {
	at := func(i int) int { return index[max(0, min(i, n))] }
	r := ct.lineGetStringRange(line)
	sl := shapedLine{start: at(r.location), end: at(r.location + r.length)}
	runs := ct.lineGetGlyphRuns(line)
	mark := len(e.shapeScratch.runs)
	for i := range ct.arrayGetCount(runs) {
		run := ct.arrayGetValueAtIndex(runs, i)
		rr := ct.runGetStringRange(run)
		attrs := ct.runGetAttributes(run)
		f := e.runFont(ct.dictionaryGetValue(attrs, ct.fontAttributeName), italic || ct.dictionaryGetValue(attrs, e.italic) != 0)
		sr := shapedRun{font: f, start: at(rr.location), end: at(rr.location + rr.length)}
		if count := ct.runGetGlyphCount(run); count > 0 {
			e.runIDs = slices.Grow(e.runIDs[:0], count)[:count]
			e.runPoints = slices.Grow(e.runPoints[:0], count)[:count]
			e.runAdvances = slices.Grow(e.runAdvances[:0], count)[:count]
			e.runIndices = slices.Grow(e.runIndices[:0], count)[:count]
			glyphs, positions, advances, indices := e.runIDs, e.runPoints, e.runAdvances, e.runIndices
			all := cfRange{}
			ct.runGetGlyphs(run, all, &glyphs[0])
			ct.runGetPositions(run, all, &positions[0])
			ct.runGetAdvances(run, all, &advances[0])
			ct.runGetStringIndices(run, all, &indices[0])
			rtl := ct.runGetStatus(run)&ctRunStatusRightToLeft != 0
			sr.glyphs = e.glyphRoom(count)
			for j := range count {
				sr.glyphs[j] = Glyph{
					Font: f, ID: uint32(glyphs[j]),
					X: float32(positions[j].x), Y: -float32(positions[j].y),
					Advance: float32(advances[j].w),
					Cluster: at(indices[j]), RTL: rtl,
				}
			}
		}
		sl.runs = e.addRun(mark, sr)
	}
	penAdvances(sl.runs)
	return sl
}

// penAdvances gives the room Core Text leaves between letters by their
// positions alone, as between Han and Latin letters in Chinese or
// Japanese text, to the advance of the letter before it, so that the
// widths of lines and the carets follow the pen. Only room ahead of the
// pen counts: a mark placed back over its letter leaves the advances as
// they are. Right-to-left runs keep theirs.
func penAdvances(runs []shapedRun) {
	var prev *Glyph
	var pen float32
	for i := range runs {
		for j := range runs[i].glyphs {
			g := &runs[i].glyphs[j]
			if prev != nil && !prev.RTL && !g.RTL {
				if gap := g.X - pen; gap > 1e-3 {
					prev.Advance += gap
					pen += gap
				}
			} else {
				pen = g.X
			}
			pen += g.Advance
			prev = g
		}
	}
}

func (e *coreText) isColor(font uintptr) bool {
	c, ok := e.color[font]
	if !ok {
		c = ct.fontGetSymbolicTraits(font)&ctFontColorGlyphsTrait != 0
		e.color[font] = c
	}
	return c
}

// positions follows Core Graphics' subpixel quantization, which AppKit
// leaves on: glyphs go to the position left of their pen, among fewer the
// larger they are, at most five a pixel and one from 34 pixels an em.
func (e *coreText) positions(f *Font, scale float32, _ bool) (int, bool) {
	px := float64(f.Size * scale)
	if px <= 0 {
		return 1, false
	}
	return min(5, max(1, int(math.Ceil(100/(3*px)-1e-9)))), false
}

// baseline is the whole pixel at or below y: Core Graphics draws a glyph
// from the pixel below its pen in its y-up space, as AppKit does on
// flipped views.
func (e *coreText) baseline(y float32) float32 { return float32(math.Ceil(float64(y) - 1e-3)) }

// textParams leaves coverage as it is: Core Graphics blends text in sRGB
// space, as renderers do, without subpixels since macOS 10.14.
func (e *coreText) textParams() (scene.TextParams, bool) { return scene.TextParams{}, false }

func (e *coreText) glyph(f *Font, id uint32, scale, dx float32, shade Shade, _ bool) bitmap {
	font := f.native
	g := uint16(id)
	var r cgRect
	ct.fontGetBoundingRects(font, ctFontOrientationDefault, &g, &r, 1)
	if r.w <= 0 || r.h <= 0 {
		return bitmap{}
	}
	s, x := float64(scale), float64(dx)
	// Smoothing spreads a glyph up to a pixel further.
	smooth := f.shaded && shade != Flat || shade == Thick
	pad := 1
	if smooth {
		pad = 2
	}
	// Core Graphics' y goes up: the box from r.y to r.y+r.h above the
	// baseline is from -(r.y+r.h) to -r.y below it.
	left := int(math.Floor(r.x*s+x)) - pad
	right := int(math.Ceil((r.x+r.w)*s+x)) + pad
	top := int(math.Floor(-(r.y+r.h)*s)) - pad
	bottom := int(math.Ceil(-r.y*s)) + pad
	w, h := right-left, bottom-top
	if w > 2048 || h > 2048 {
		return bitmap{}
	}
	color := e.isColor(font)
	var pix []byte
	var ctx uintptr
	if color {
		pix = make([]byte, 4*w*h)
		ctx = ct.bitmapContextCreate(unsafe.Pointer(&pix[0]), w, h, 8, 4*w, e.srgb, cgImageAlphaPremultipliedLst|cgBitmapByteOrder32Big)
	} else {
		pix = make([]byte, w*h)
		ctx = ct.bitmapContextCreate(unsafe.Pointer(&pix[0]), w, h, 8, w, 0, cgImageAlphaOnly)
	}
	if ctx == 0 {
		return bitmap{}
	}
	ct.contextAntialias(ctx, true)
	// Font smoothing emboldens glyphs more the lighter the fill color is,
	// even in a context of alpha alone: the gray of the shade sets how
	// much. Fonts are shaded while the user leaves it on; thick text is
	// smoothed with white, the most.
	ct.contextSmoothFonts(ctx, smooth)
	ct.contextAllowSubpixelPos(ctx, true)
	ct.contextSubpixelPos(ctx, true)
	ct.contextAllowQuantize(ctx, false)
	ct.contextQuantize(ctx, false)
	fill := 0.0
	if shade == Thick {
		fill = 1
	} else if smooth {
		fill = shadeGray(shade)
	}
	ct.contextSetFill(ctx, fill, fill, fill, 1)
	ct.contextScaleCTM(ctx, s, s)
	// The bitmap's rows go from its top; the context's origin is at its
	// bottom left, the baseline bottom pixels above it.
	pos := cgPoint{(x - float64(left)) / s, float64(bottom) / s}
	ct.fontDrawGlyphs(font, &g, &pos, 1, ctx)
	ct.contextRelease(ctx)
	runtime.KeepAlive(pix)
	return bitmap{left: left, top: top, w: w, h: h, pix: pix, color: color}
}

// decorate places underlines and strikethroughs as Core Text draws them
// for AppKit, which snaps them in points, the DIPs of a screen, from the
// baseline: an underline as low and thick as those of the fonts of the
// glyphs under it want it, skipping their ink, a strikethrough through the
// middle of the x-height of the text's own font, and both from where the
// text starts to its width rounded up to a whole point, leaving out
// whitespace starting or ending a line. The baseline is the glyphs', a
// whole pixel, as AppKit's labels have on whole points.
func (e *coreText) decorate(r decoRange) []Stroke {
	gs := r.line.Glyphs
	i, j := r.i, r.j
	if i == 0 {
		for i < j && unicode.IsSpace(r.rune(i)) {
			i++
		}
	}
	if j == len(gs) {
		for j > i && unicode.IsSpace(r.rune(j-1)) {
			j--
		}
	}
	if i >= j {
		return nil
	}
	x0, _ := r.span(i, j)
	var width float32
	for _, g := range gs[i:j] {
		width += g.Advance
	}
	x1 := x0 + float32(math.Ceil(float64(width)-1e-4))
	var center, thick float32 // below the baseline
	if r.d == Strikethrough {
		p, t := ctStrikethrough(r.font)
		center, thick = -p, t
	} else {
		r.runs(func(f *Font, _, _ int) {
			p, t := ctUnderline(f)
			center, thick = max(center, p), max(thick, t)
		})
	}
	pieces := [][2]float32{{x0, x1}}
	if r.d == Underline {
		var cuts [][2]float32
		for k := i; k < j; k++ {
			g := gs[k]
			if g.Font == nil || !skipsInk(r.rune(k)) {
				continue
			}
			if a, b, ok := e.inkInBand(g.Font, g.ID, -(center + thick/2), -(center - thick/2)); ok {
				cuts = append(cuts, [2]float32{g.X + a - thick, g.X + b + thick})
			}
		}
		pieces = cut(x0, x1, cuts, 0.75*thick)
	}
	s := r.scale
	base := e.baseline(r.baseline() * s)
	out := make([]Stroke, len(pieces))
	for k, p := range pieces {
		out[k] = Stroke{X0: (r.x + p[0]) * s, X1: (r.x + p[1]) * s, Top: base + (center-thick/2)*s, Bottom: base + (center+thick/2)*s}
	}
	return out
}

// ctUnderline returns where Core Text centers the underline of a font
// below the baseline, and how thick it draws it, in points.
func ctUnderline(f *Font) (center, thick float32) {
	font := f.native
	pos, th := ct.fontGetUnderlinePos(font), ct.fontGetUnderlineThick(font)
	asc, desc := ct.fontGetAscent(font), ct.fontGetDescent(font)
	d := desc
	if desc < 2 {
		d = (asc + desc) / 4
	}
	reach := min(d*5.3636991028295373, asc+desc)
	if pos >= 0 { // no underline position
		pos = -0.08805546253922189 * reach
	}
	p, t := -pos, th
	if d >= 2 && th > 0.35 {
		t = math.Ceil(th)
		if t >= d || d <= 4 && t >= 3 || d <= 2.5 && t >= 2 {
			t--
		}
		p = snapLine(p, t)
		if p < 1.5 || p == 1.5 && d > 4 {
			p++
		}
	}
	if d > 0 {
		p = min(p, math.Floor(d)-t/2)
	}
	p = max(p, math.Ceil(th)+t/2)
	if t <= 0 { // no underline thickness
		t = reach * 0.044027731269610945
	}
	return float32(p), float32(t)
}

// ctStrikethrough returns where Core Text centers the strikethrough of a
// font above the baseline, and how thick it draws it, in points: through
// the middle of the x-height, as thick as the underline.
func ctStrikethrough(f *Font) (center, thick float32) {
	font := f.native
	p, th := ct.fontGetXHeight(font)/2, ct.fontGetUnderlineThick(font)
	if p <= 0 {
		// No x-height: the strikeout of the OS/2 table.
		if t := ct.fontCopyTable(font, 0x4F532F32, 0); t != 0 {
			if ct.dataGetLength(t) >= 30 {
				b := unsafe.Slice(ct.dataGetBytePtr(t), 30)
				size, pos := int16(uint16(b[26])<<8|uint16(b[27])), int16(uint16(b[28])<<8|uint16(b[29]))
				em := ct.fontGetSize(font) / float64(max(ct.fontGetUnitsPerEm(font), 1))
				p, th = (float64(pos)-float64(size)/2)*em, float64(size)*em
			}
			ct.release(t)
		}
	}
	t := th
	if p > 1 && th > 0.35 {
		t = math.Ceil(th)
		p = snapLine(p, t)
	}
	return float32(p), float32(t)
}

// join: Core Graphics blends each glyph of a run in turn.
func (e *coreText) join() bool { return false }

// snapLine puts a line of thickness t (whole points) centered at p on
// whole points.
func snapLine(p, t float64) float64 {
	if int(t)%2 == 0 {
		return math.Floor(p + 0.5)
	}
	return math.Floor(p) + 0.5
}

type bandKey struct {
	font   uintptr
	glyph  uint32
	lo, hi float32
}

// inkInBand returns how far left and right of its origin the outline of a
// glyph reaches between lo and hi points above the baseline.
func (e *coreText) inkInBand(f *Font, id uint32, lo, hi float32) (a, b float32, ok bool) {
	key := bandKey{f.native, id, lo, hi}
	if v, seen := e.bands[key]; seen {
		return v[0], v[1], v[0] <= v[1]
	}
	if len(e.bands) >= 4096 {
		clear(e.bands)
	}
	segs := glyphOutline(f.native, uint16(id))
	ax, bx, found := bandX(segs, float64(lo), float64(hi))
	v := [2]float32{1, 0}
	if found {
		v = [2]float32{float32(ax), float32(bx)}
	}
	e.bands[key] = v
	return v[0], v[1], found
}

// The outline of a glyph, flattened into segments, as CGPathApply hands
// its elements to the one callback made for it.
var (
	outlineOnce sync.Once
	outlineFn   uintptr
	outlineSegs [][4]float64
	outlineAt   [2]float64 // the current point
	outlineFrom [2]float64 // the start of the subpath
)

type cgPathElement struct {
	kind   int32
	points *cgPoint
}

// glyphOutline returns the outline of a glyph in points, y up, as line
// segments. The text system's lock serializes its callers.
func glyphOutline(font uintptr, glyph uint16) [][4]float64 {
	outlineOnce.Do(func() {
		outlineFn = purego.NewCallback(func(_ uintptr, el *cgPathElement) uintptr {
			pt := func(i int) [2]float64 {
				p := unsafe.Slice(el.points, i+1)[i]
				return [2]float64{p.x, p.y}
			}
			line := func(q [2]float64) {
				outlineSegs = append(outlineSegs, [4]float64{outlineAt[0], outlineAt[1], q[0], q[1]})
				outlineAt = q
			}
			switch el.kind {
			case 0: // move
				outlineAt = pt(0)
				outlineFrom = outlineAt
			case 1: // line
				line(pt(0))
			case 2, 3: // quadratic and cubic curves, in 16 lines
				p0 := outlineAt
				c := []([2]float64){p0, pt(0), pt(1)}
				if el.kind == 3 {
					c = append(c, pt(2))
				}
				for k := 1; k <= 16; k++ {
					t := float64(k) / 16
					line(bezier(c, t))
				}
			case 4: // close
				line(outlineFrom)
			}
			return 0
		})
	})
	outlineSegs = outlineSegs[:0]
	path := ct.fontCreatePath(font, glyph, 0)
	if path == 0 {
		return nil
	}
	ct.pathApply(path, 0, outlineFn)
	ct.pathRelease(path)
	return slices.Clone(outlineSegs)
}

// bezier returns the point at t of a Bézier curve of control points c.
func bezier(c [][2]float64, t float64) [2]float64 {
	p := slices.Clone(c)
	for n := len(p) - 1; n > 0; n-- {
		for k := range n {
			p[k] = [2]float64{p[k][0] + (p[k+1][0]-p[k][0])*t, p[k][1] + (p[k+1][1]-p[k][1])*t}
		}
	}
	return p[0]
}

// shadeGray returns the sRGB gray of the relative luminance a shade
// stands for.
func shadeGray(shade Shade) float64 {
	y := float64(shade) / Shades
	if y <= 0.0031308 {
		return y * 12.92
	}
	return 1.055*math.Pow(y, 1/2.4) - 0.055
}

// appFamilies lists the families of the app's fonts, monospaced when
// a registered face's symbolic traits say so.
func (e *coreText) appFamilies() []FontFamily {
	var out []FontFamily
	for _, name := range e.names {
		mono := slices.ContainsFunc(e.registered[strings.ToLower(name)], func(f registeredFace) bool { return f.mono })
		out = append(out, FontFamily{Name: name, Monospaced: mono})
	}
	return out
}

// systemFamilies lists the font manager's families, which hold none of the
// app's, monospaced when their regular font's symbolic traits say so: Core
// Text alone, which is thread-safe, without the engine's state.
func (e *coreText) systemFamilies() func() []FontFamily {
	return func() []FontFamily {
		arr := ct.availableFamilyNames()
		if arr == 0 {
			return nil
		}
		defer ct.release(arr)
		var out []FontFamily
		for i := range ct.arrayGetCount(arr) {
			name := goString(ct.arrayGetValueAtIndex(arr, i))
			if name == "" || strings.HasPrefix(name, ".") {
				continue // the system's hidden families
			}
			out = append(out, FontFamily{Name: name, Monospaced: monospacedFamily(name)})
		}
		return out
	}
}

// monospacedFamily reports whether a family's regular font is monospaced.
func monospacedFamily(family string) bool {
	desc := namedDesc(family, 400, false)
	if desc == 0 {
		return false
	}
	defer ct.release(desc)
	traits := ct.descriptorAttribute(desc, ct.traitsAttribute)
	if traits == 0 {
		return false
	}
	defer ct.release(traits)
	s := ct.dictionaryGetValue(traits, ct.symbolicTrait)
	if s == 0 {
		return false
	}
	var v int64
	ct.numberGetValue(s, cfNumberSInt64Type, unsafe.Pointer(&v))
	return v&ctFontMonoSpaceTrait != 0
}

func (e *coreText) register(data []byte, family string) error {
	if ct.descriptorsFromData == nil {
		return errors.New("mygo: adding fonts needs macOS 10.13 or later")
	}
	if len(data) == 0 {
		return errors.New("mygo: no font data")
	}
	d := ct.dataCreate(0, &data[0], len(data))
	defer ct.release(d)
	descs := ct.descriptorsFromData(d)
	if descs == 0 {
		return errors.New("mygo: cannot add font: not a font Core Text reads")
	}
	defer ct.release(descs)
	n := ct.arrayGetCount(descs)
	if n == 0 {
		return errors.New("mygo: cannot add font: not a font Core Text reads")
	}
	for i := range n {
		desc := ct.arrayGetValueAtIndex(descs, i)
		face := registeredFace{desc: ct.retain(desc)}
		var name string
		if s := ct.descriptorAttribute(desc, ct.familyNameAttribute); s != 0 {
			name = goString(s)
			ct.release(s)
		}
		if traits := ct.descriptorAttribute(desc, ct.traitsAttribute); traits != 0 {
			if w := ct.dictionaryGetValue(traits, ct.weightTrait); w != 0 {
				ct.numberGetValue(w, cfNumberFloat64Type, unsafe.Pointer(&face.weight))
			}
			if s := ct.dictionaryGetValue(traits, ct.symbolicTrait); s != 0 {
				var v int64
				ct.numberGetValue(s, cfNumberSInt64Type, unsafe.Pointer(&v))
				face.italic = v&ctFontItalicTrait != 0
			}
			ct.release(traits)
		}
		if font := ct.fontWithDescriptor(desc, 12, 0); font != 0 {
			face.mono = ct.fontGetSymbolicTraits(font)&ctFontMonoSpaceTrait != 0
			ct.release(font)
		}
		for _, f := range []string{name, family} {
			if f != "" {
				key := strings.ToLower(f)
				e.registered[key] = append(e.registered[key], face)
			}
		}
		// Listed under the name the app gave it, which is the one it
		// uses, or its own.
		e.names = append(e.names, cmp.Or(family, name))
	}
	e.forgetPrimary()
	return nil
}

// forgetPrimary lets go of the fonts of styles, and of their attributes.
func (e *coreText) forgetPrimary() {
	for _, f := range e.primary {
		ct.release(f)
	}
	clear(e.primary)
	for _, a := range e.attrs {
		ct.release(a)
	}
	clear(e.attrs)
}

func (e *coreText) fontCount() int {
	n := 0
	for _, fs := range e.fonts {
		n += len(fs)
	}
	return n
}

// forgetFonts lets go of the CTFonts of the Fonts made so far, and of
// what is known of them; the fonts of styles stay.
func (e *coreText) forgetFonts() {
	for _, fs := range e.fonts {
		for _, f := range fs {
			ct.release(f.native)
		}
	}
	clear(e.fonts)
	clear(e.native)
	clear(e.oblique)
	clear(e.color)
}
