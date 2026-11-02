import AppKit
import Carbon
import IOKit.hidsystem

nonisolated extension HotkeyPreference {
    static func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            AppPreferenceKey.hotkeyInputType: Hotkey.Input.Kind.keyboard.rawValue,
            AppPreferenceKey.hotkeyKeyCode: Int(defaultKeyCode),
            AppPreferenceKey.hotkeyMouseButtonNumber: middleMouseButtonNumber,
            AppPreferenceKey.hotkeyModifiers: Int(defaultModifiers.rawValue),
            AppPreferenceKey.hotkeySidedModifiers: 0,
            AppPreferenceKey.translationHotkeyInputType: Hotkey.Input.Kind.keyboard.rawValue,
            AppPreferenceKey.translationHotkeyKeyCode: Int(defaultTranslationKeyCode),
            AppPreferenceKey.translationHotkeyMouseButtonNumber: middleMouseButtonNumber,
            AppPreferenceKey.translationHotkeyModifiers: Int(defaultTranslationModifiers.rawValue),
            AppPreferenceKey.translationHotkeySidedModifiers: 0,
            AppPreferenceKey.rewriteHotkeyInputType: Hotkey.Input.Kind.keyboard.rawValue,
            AppPreferenceKey.rewriteHotkeyKeyCode: Int(defaultRewriteKeyCode),
            AppPreferenceKey.rewriteHotkeyMouseButtonNumber: middleMouseButtonNumber,
            AppPreferenceKey.rewriteHotkeyModifiers: Int(defaultRewriteModifiers.rawValue),
            AppPreferenceKey.rewriteHotkeySidedModifiers: 0,
            AppPreferenceKey.meetingHotkeyInputType: Hotkey.Input.Kind.keyboard.rawValue,
            AppPreferenceKey.meetingHotkeyKeyCode: Int(defaultMeetingKeyCode),
            AppPreferenceKey.meetingHotkeyMouseButtonNumber: middleMouseButtonNumber,
            AppPreferenceKey.meetingHotkeyModifiers: Int(defaultMeetingModifiers.rawValue),
            AppPreferenceKey.meetingHotkeySidedModifiers: 0,
            AppPreferenceKey.customPasteHotkeyInputType: Hotkey.Input.Kind.keyboard.rawValue,
            AppPreferenceKey.customPasteHotkeyKeyCode: Int(defaultCustomPasteKeyCode),
            AppPreferenceKey.customPasteHotkeyMouseButtonNumber: middleMouseButtonNumber,
            AppPreferenceKey.customPasteHotkeyModifiers: Int(defaultCustomPasteModifiers.rawValue),
            AppPreferenceKey.customPasteHotkeySidedModifiers: 0,
            AppPreferenceKey.hotkeyTriggerMode: defaultTriggerMode.rawValue,
            AppPreferenceKey.rewriteHotkeyActivationMode: defaultRewriteActivationMode.rawValue,
            AppPreferenceKey.hotkeyDistinguishModifierSides: defaultDistinguishModifierSides,
            AppPreferenceKey.hotkeyPreset: defaultPreset.rawValue,
            AppPreferenceKey.hotkeyCaptureInProgress: false,
        ])
        migrateHotkeyBindingsIfNeeded()
    }

    static func migrateDefaultsIfNeeded() {
        let defaults = UserDefaults.standard
        guard let keyCodeValue = defaults.object(forKey: AppPreferenceKey.hotkeyKeyCode) as? Int,
              let modifiersValue = defaults.object(forKey: AppPreferenceKey.hotkeyModifiers) as? Int
        else {
            syncStoredPresetValuesIfNeeded()
            return
        }

        let keyCode = UInt16(exactly: keyCodeValue) ?? defaultKeyCode
        let modifiers = NSEvent.ModifierFlags(rawValue: UInt(exactly: modifiersValue) ?? defaultModifiers.rawValue).intersection(.hotkeyRelevant)

        if keyCode == modifierOnlyKeyCode && modifiers == [.control, .option] {
            save(keyCode: defaultKeyCode, modifiers: defaultModifiers, sidedModifiers: [])
        }

        syncStoredPresetValuesIfNeeded()
        migrateHotkeyBindingsIfNeeded()
    }

    static func migrateHotkeyBindingsIfNeeded(defaults: UserDefaults = .standard) {
        migrateBindingsIfNeeded(
            bindingsKey: AppPreferenceKey.transcriptionHotkeyBindings,
            legacyHotkey: legacyHotkey(
                inputTypeKey: AppPreferenceKey.hotkeyInputType,
                keyCodeKey: AppPreferenceKey.hotkeyKeyCode,
                mouseButtonKey: AppPreferenceKey.hotkeyMouseButtonNumber,
                modifiersKey: AppPreferenceKey.hotkeyModifiers,
                sidedModifiersKey: AppPreferenceKey.hotkeySidedModifiers,
                defaultKeyCode: defaultKeyCode,
                defaultModifiers: defaultModifiers,
                defaults: defaults
            ),
            defaultBehavior: legacyDefaultBehavior(defaults: defaults),
            defaults: defaults
        )
        migrateBindingsIfNeeded(
            bindingsKey: AppPreferenceKey.translationHotkeyBindings,
            legacyHotkey: legacyHotkey(
                inputTypeKey: AppPreferenceKey.translationHotkeyInputType,
                keyCodeKey: AppPreferenceKey.translationHotkeyKeyCode,
                mouseButtonKey: AppPreferenceKey.translationHotkeyMouseButtonNumber,
                modifiersKey: AppPreferenceKey.translationHotkeyModifiers,
                sidedModifiersKey: AppPreferenceKey.translationHotkeySidedModifiers,
                defaultKeyCode: defaultTranslationKeyCode,
                defaultModifiers: defaultTranslationModifiers,
                defaults: defaults
            ),
            defaultBehavior: legacyDefaultBehavior(defaults: defaults),
            defaults: defaults
        )
        migrateBindingsIfNeeded(
            bindingsKey: AppPreferenceKey.meetingHotkeyBindings,
            legacyHotkey: legacyHotkey(
                inputTypeKey: AppPreferenceKey.meetingHotkeyInputType,
                keyCodeKey: AppPreferenceKey.meetingHotkeyKeyCode,
                mouseButtonKey: AppPreferenceKey.meetingHotkeyMouseButtonNumber,
                modifiersKey: AppPreferenceKey.meetingHotkeyModifiers,
                sidedModifiersKey: AppPreferenceKey.meetingHotkeySidedModifiers,
                defaultKeyCode: defaultMeetingKeyCode,
                defaultModifiers: defaultMeetingModifiers,
                defaults: defaults
            ),
            defaultBehavior: legacyDefaultBehavior(defaults: defaults),
            defaults: defaults
        )

        guard defaults.object(forKey: AppPreferenceKey.rewriteHotkeyBindings) == nil else { return }
        let rewriteActivationMode = loadRewriteActivationMode(defaults: defaults)
        let rewriteHotkey = rewriteActivationMode == .doubleTapTranscriptionHotkey
            ? legacyHotkey(
                inputTypeKey: AppPreferenceKey.hotkeyInputType,
                keyCodeKey: AppPreferenceKey.hotkeyKeyCode,
                mouseButtonKey: AppPreferenceKey.hotkeyMouseButtonNumber,
                modifiersKey: AppPreferenceKey.hotkeyModifiers,
                sidedModifiersKey: AppPreferenceKey.hotkeySidedModifiers,
                defaultKeyCode: defaultKeyCode,
                defaultModifiers: defaultModifiers,
                defaults: defaults
            )
            : legacyHotkey(
                inputTypeKey: AppPreferenceKey.rewriteHotkeyInputType,
                keyCodeKey: AppPreferenceKey.rewriteHotkeyKeyCode,
                mouseButtonKey: AppPreferenceKey.rewriteHotkeyMouseButtonNumber,
                modifiersKey: AppPreferenceKey.rewriteHotkeyModifiers,
                sidedModifiersKey: AppPreferenceKey.rewriteHotkeySidedModifiers,
                defaultKeyCode: defaultRewriteKeyCode,
                defaultModifiers: defaultRewriteModifiers,
                defaults: defaults
            )
        let rewriteBehavior: TriggerBehavior = rewriteActivationMode == .doubleTapTranscriptionHotkey
            ? .doubleTap
            : legacyDefaultBehavior(defaults: defaults)
        saveBindings(
            [.init(hotkey: rewriteHotkey, behavior: rewriteBehavior)],
            forKey: AppPreferenceKey.rewriteHotkeyBindings,
            defaults: defaults
        )
    }

    static func loadTranscriptionBindings(defaults: UserDefaults = .standard) -> [HotkeyBinding] {
        loadBindings(
            forKey: AppPreferenceKey.transcriptionHotkeyBindings,
            fallbackHotkey: load(),
            defaults: defaults
        )
    }

    static func loadTranslationBindings(defaults: UserDefaults = .standard) -> [HotkeyBinding] {
        loadBindings(
            forKey: AppPreferenceKey.translationHotkeyBindings,
            fallbackHotkey: loadTranslation(),
            defaults: defaults
        )
    }

    static func loadMeetingBindings(defaults: UserDefaults = .standard) -> [HotkeyBinding] {
        loadBindings(
            forKey: AppPreferenceKey.meetingHotkeyBindings,
            fallbackHotkey: loadMeeting(),
            defaults: defaults
        )
    }

    static func loadRewriteBindings(defaults: UserDefaults = .standard) -> [HotkeyBinding] {
        loadBindings(
            forKey: AppPreferenceKey.rewriteHotkeyBindings,
            fallbackHotkey: loadRewrite(),
            defaults: defaults
        )
    }

    static func loadNoteBindings(defaults: UserDefaults = .standard) -> [HotkeyBinding] {
        let fallback = Hotkey(
            keyCode: defaultNoteKeyCode,
            modifiers: defaultNoteModifiers,
            sidedModifiers: defaultNoteSidedModifiers
        )
        let loaded = loadBindings(
            forKey: AppPreferenceKey.noteHotkeyBindings,
            fallbackHotkey: fallback,
            defaults: defaults
        )
        guard !loaded.isEmpty else {
            return [.init(hotkey: fallback, behavior: .tap)]
        }
        return loaded.map {
            .init(id: $0.id, hotkey: $0.hotkey, behavior: .tap)
        }
    }

    static func saveTranscriptionBindings(_ bindings: [HotkeyBinding], defaults: UserDefaults = .standard) {
        saveBindings(bindings, forKey: AppPreferenceKey.transcriptionHotkeyBindings, defaults: defaults)
        if let first = sanitizedBindings(bindings).first {
            save(first.hotkey, defaults: defaults, syncBindings: false)
        }
    }

    static func saveTranslationBindings(_ bindings: [HotkeyBinding], defaults: UserDefaults = .standard) {
        saveBindings(bindings, forKey: AppPreferenceKey.translationHotkeyBindings, defaults: defaults)
        if let first = sanitizedBindings(bindings).first {
            saveTranslation(first.hotkey, defaults: defaults, syncBindings: false)
        }
    }

    static func saveMeetingBindings(_ bindings: [HotkeyBinding], defaults: UserDefaults = .standard) {
        saveBindings(bindings, forKey: AppPreferenceKey.meetingHotkeyBindings, defaults: defaults)
        if let first = sanitizedBindings(bindings).first {
            saveMeeting(first.hotkey, defaults: defaults, syncBindings: false)
        }
    }

    static func saveRewriteBindings(_ bindings: [HotkeyBinding], defaults: UserDefaults = .standard) {
        saveBindings(bindings, forKey: AppPreferenceKey.rewriteHotkeyBindings, defaults: defaults)
        if let first = sanitizedBindings(bindings).first {
            saveRewrite(first.hotkey, defaults: defaults, syncBindings: false)
        }
    }

    static func saveNoteBindings(_ bindings: [HotkeyBinding], defaults: UserDefaults = .standard) {
        let fallback = Hotkey(
            keyCode: defaultNoteKeyCode,
            modifiers: defaultNoteModifiers,
            sidedModifiers: defaultNoteSidedModifiers
        )
        let sources = bindings.isEmpty
            ? [.init(hotkey: fallback, behavior: .tap)]
            : bindings
        saveBindings(
            sources.map { .init(id: $0.id, hotkey: $0.hotkey, behavior: .tap) },
            forKey: AppPreferenceKey.noteHotkeyBindings,
            defaults: defaults
        )
    }

    static func load() -> Hotkey {
        if let presetHotkey = resolvedPresetHotkeys()?.transcription {
            return presetHotkey
        }
        return load(
            inputTypeKey: AppPreferenceKey.hotkeyInputType,
            keyCodeKey: AppPreferenceKey.hotkeyKeyCode,
            mouseButtonKey: AppPreferenceKey.hotkeyMouseButtonNumber,
            modifiersKey: AppPreferenceKey.hotkeyModifiers,
            sidedModifiersKey: AppPreferenceKey.hotkeySidedModifiers,
            defaultKeyCode: defaultKeyCode,
            defaultModifiers: defaultModifiers
        )
    }

    static func save(keyCode: UInt16, modifiers: NSEvent.ModifierFlags, sidedModifiers: SidedModifierFlags) {
        save(.init(keyCode: keyCode, modifiers: modifiers, sidedModifiers: sidedModifiers))
    }

    static func save(_ hotkey: Hotkey, defaults: UserDefaults = .standard, syncBindings: Bool = true) {
        save(
            hotkey,
            inputTypeKey: AppPreferenceKey.hotkeyInputType,
            keyCodeKey: AppPreferenceKey.hotkeyKeyCode,
            mouseButtonKey: AppPreferenceKey.hotkeyMouseButtonNumber,
            modifiersKey: AppPreferenceKey.hotkeyModifiers,
            sidedModifiersKey: AppPreferenceKey.hotkeySidedModifiers,
            defaults: defaults
        )
        if syncBindings {
            saveBindings(
                [.init(hotkey: hotkey, behavior: legacyDefaultBehavior(defaults: defaults))],
                forKey: AppPreferenceKey.transcriptionHotkeyBindings,
                defaults: defaults
            )
        }
    }

    static func loadTranslation() -> Hotkey {
        if let presetHotkey = resolvedPresetHotkeys()?.translation {
            return presetHotkey
        }
        return load(
            inputTypeKey: AppPreferenceKey.translationHotkeyInputType,
            keyCodeKey: AppPreferenceKey.translationHotkeyKeyCode,
            mouseButtonKey: AppPreferenceKey.translationHotkeyMouseButtonNumber,
            modifiersKey: AppPreferenceKey.translationHotkeyModifiers,
            sidedModifiersKey: AppPreferenceKey.translationHotkeySidedModifiers,
            defaultKeyCode: defaultTranslationKeyCode,
            defaultModifiers: defaultTranslationModifiers
        )
    }

    static func saveTranslation(keyCode: UInt16, modifiers: NSEvent.ModifierFlags, sidedModifiers: SidedModifierFlags) {
        saveTranslation(.init(keyCode: keyCode, modifiers: modifiers, sidedModifiers: sidedModifiers))
    }

    static func saveTranslation(_ hotkey: Hotkey, defaults: UserDefaults = .standard, syncBindings: Bool = true) {
        save(
            hotkey,
            inputTypeKey: AppPreferenceKey.translationHotkeyInputType,
            keyCodeKey: AppPreferenceKey.translationHotkeyKeyCode,
            mouseButtonKey: AppPreferenceKey.translationHotkeyMouseButtonNumber,
            modifiersKey: AppPreferenceKey.translationHotkeyModifiers,
            sidedModifiersKey: AppPreferenceKey.translationHotkeySidedModifiers,
            defaults: defaults
        )
        if syncBindings {
            saveBindings(
                [.init(hotkey: hotkey, behavior: legacyDefaultBehavior(defaults: defaults))],
                forKey: AppPreferenceKey.translationHotkeyBindings,
                defaults: defaults
            )
        }
    }

    static func loadRewrite() -> Hotkey {
        if let presetHotkey = resolvedPresetHotkeys()?.rewrite {
            return presetHotkey
        }
        return load(
            inputTypeKey: AppPreferenceKey.rewriteHotkeyInputType,
            keyCodeKey: AppPreferenceKey.rewriteHotkeyKeyCode,
            mouseButtonKey: AppPreferenceKey.rewriteHotkeyMouseButtonNumber,
            modifiersKey: AppPreferenceKey.rewriteHotkeyModifiers,
            sidedModifiersKey: AppPreferenceKey.rewriteHotkeySidedModifiers,
            defaultKeyCode: defaultRewriteKeyCode,
            defaultModifiers: defaultRewriteModifiers
        )
    }

    static func saveRewrite(keyCode: UInt16, modifiers: NSEvent.ModifierFlags, sidedModifiers: SidedModifierFlags) {
        saveRewrite(.init(keyCode: keyCode, modifiers: modifiers, sidedModifiers: sidedModifiers))
    }

    static func saveRewrite(_ hotkey: Hotkey, defaults: UserDefaults = .standard, syncBindings: Bool = true) {
        save(
            hotkey,
            inputTypeKey: AppPreferenceKey.rewriteHotkeyInputType,
            keyCodeKey: AppPreferenceKey.rewriteHotkeyKeyCode,
            mouseButtonKey: AppPreferenceKey.rewriteHotkeyMouseButtonNumber,
            modifiersKey: AppPreferenceKey.rewriteHotkeyModifiers,
            sidedModifiersKey: AppPreferenceKey.rewriteHotkeySidedModifiers,
            defaults: defaults
        )
        if syncBindings {
            let behavior: TriggerBehavior = loadRewriteActivationMode(defaults: defaults) == .doubleTapTranscriptionHotkey
                ? .doubleTap
                : legacyDefaultBehavior(defaults: defaults)
            saveBindings(
                [.init(hotkey: hotkey, behavior: behavior)],
                forKey: AppPreferenceKey.rewriteHotkeyBindings,
                defaults: defaults
            )
        }
    }

    static func loadMeeting() -> Hotkey {
        if let presetHotkey = resolvedPresetHotkeys()?.meeting {
            return presetHotkey
        }
        return load(
            inputTypeKey: AppPreferenceKey.meetingHotkeyInputType,
            keyCodeKey: AppPreferenceKey.meetingHotkeyKeyCode,
            mouseButtonKey: AppPreferenceKey.meetingHotkeyMouseButtonNumber,
            modifiersKey: AppPreferenceKey.meetingHotkeyModifiers,
            sidedModifiersKey: AppPreferenceKey.meetingHotkeySidedModifiers,
            defaultKeyCode: defaultMeetingKeyCode,
            defaultModifiers: defaultMeetingModifiers
        )
    }

    static func saveMeeting(keyCode: UInt16, modifiers: NSEvent.ModifierFlags, sidedModifiers: SidedModifierFlags) {
        saveMeeting(.init(keyCode: keyCode, modifiers: modifiers, sidedModifiers: sidedModifiers))
    }

    static func saveMeeting(_ hotkey: Hotkey, defaults: UserDefaults = .standard, syncBindings: Bool = true) {
        save(
            hotkey,
            inputTypeKey: AppPreferenceKey.meetingHotkeyInputType,
            keyCodeKey: AppPreferenceKey.meetingHotkeyKeyCode,
            mouseButtonKey: AppPreferenceKey.meetingHotkeyMouseButtonNumber,
            modifiersKey: AppPreferenceKey.meetingHotkeyModifiers,
            sidedModifiersKey: AppPreferenceKey.meetingHotkeySidedModifiers,
            defaults: defaults
        )
        if syncBindings {
            saveBindings(
                [.init(hotkey: hotkey, behavior: legacyDefaultBehavior(defaults: defaults))],
                forKey: AppPreferenceKey.meetingHotkeyBindings,
                defaults: defaults
            )
        }
    }

    static func loadCustomPaste() -> Hotkey {
        if let presetHotkey = resolvedPresetHotkeys()?.customPaste {
            return presetHotkey
        }
        return normalizeCustomPasteHotkey(load(
            inputTypeKey: AppPreferenceKey.customPasteHotkeyInputType,
            keyCodeKey: AppPreferenceKey.customPasteHotkeyKeyCode,
            mouseButtonKey: AppPreferenceKey.customPasteHotkeyMouseButtonNumber,
            modifiersKey: AppPreferenceKey.customPasteHotkeyModifiers,
            sidedModifiersKey: AppPreferenceKey.customPasteHotkeySidedModifiers,
            defaultKeyCode: defaultCustomPasteKeyCode,
            defaultModifiers: defaultCustomPasteModifiers
        ))
    }

    static func saveCustomPaste(keyCode: UInt16, modifiers: NSEvent.ModifierFlags, sidedModifiers: SidedModifierFlags) {
        saveCustomPaste(.init(keyCode: keyCode, modifiers: modifiers, sidedModifiers: sidedModifiers))
    }

    static func saveCustomPaste(_ hotkey: Hotkey, defaults: UserDefaults = .standard) {
        save(
            hotkey,
            inputTypeKey: AppPreferenceKey.customPasteHotkeyInputType,
            keyCodeKey: AppPreferenceKey.customPasteHotkeyKeyCode,
            mouseButtonKey: AppPreferenceKey.customPasteHotkeyMouseButtonNumber,
            modifiersKey: AppPreferenceKey.customPasteHotkeyModifiers,
            sidedModifiersKey: AppPreferenceKey.customPasteHotkeySidedModifiers,
            defaults: defaults
        )
    }

    static func loadTriggerMode(defaults: UserDefaults = .standard) -> TriggerMode {
        let raw = defaults.string(forKey: AppPreferenceKey.hotkeyTriggerMode)
        let requestedMode = TriggerMode(rawValue: raw ?? "") ?? defaultTriggerMode
        return enforcedTriggerMode(requestedMode, rewriteActivationMode: loadRewriteActivationMode(defaults: defaults))
    }

    static func saveTriggerMode(_ mode: TriggerMode, defaults: UserDefaults = .standard) {
        let enforcedMode = enforcedTriggerMode(mode, rewriteActivationMode: loadRewriteActivationMode(defaults: defaults))
        defaults.set(enforcedMode.rawValue, forKey: AppPreferenceKey.hotkeyTriggerMode)
    }

    static func loadRewriteActivationMode(defaults: UserDefaults = .standard) -> RewriteActivationMode {
        let raw = defaults.string(forKey: AppPreferenceKey.rewriteHotkeyActivationMode)
        return RewriteActivationMode(rawValue: raw ?? "") ?? defaultRewriteActivationMode
    }

    static func saveRewriteActivationMode(_ mode: RewriteActivationMode, defaults: UserDefaults = .standard) {
        defaults.set(mode.rawValue, forKey: AppPreferenceKey.rewriteHotkeyActivationMode)
        let currentTriggerMode = TriggerMode(
            rawValue: defaults.string(forKey: AppPreferenceKey.hotkeyTriggerMode) ?? ""
        ) ?? defaultTriggerMode
        saveTriggerMode(currentTriggerMode, defaults: defaults)
    }

    static func enforcedTriggerMode(
        _ mode: TriggerMode,
        rewriteActivationMode: RewriteActivationMode
    ) -> TriggerMode {
        rewriteActivationMode == .doubleTapTranscriptionHotkey ? .tap : mode
    }

    static func loadDistinguishModifierSides() -> Bool {
        true
    }

    static func loadPreset() -> Preset {
        let raw = UserDefaults.standard.string(forKey: AppPreferenceKey.hotkeyPreset)
        return Preset(rawValue: raw ?? "") ?? defaultPreset
    }

    @discardableResult
    static func applyPreset(_ preset: Preset) -> PresetHotkeys? {
        UserDefaults.standard.set(preset.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
        guard let values = presetHotkeys(for: preset) else { return nil }
        applyPresetHotkeys(values)
        return values
    }

    private static func resolvedPresetHotkeys() -> PresetHotkeys? {
        let preset = loadPreset()
        guard preset != .custom else { return nil }
        return presetHotkeys(for: preset)
    }

    private static func syncStoredPresetValuesIfNeeded() {
        guard let presetValues = resolvedPresetHotkeys() else { return }
        applyPresetHotkeys(presetValues)
    }

    private static func applyPresetHotkeys(_ presetValues: PresetHotkeys) {
        UserDefaults.standard.set(presetValues.distinguishSides, forKey: AppPreferenceKey.hotkeyDistinguishModifierSides)
        save(presetValues.transcription, syncBindings: false)
        saveTranslation(presetValues.translation, syncBindings: false)
        saveRewrite(presetValues.rewrite, syncBindings: false)
        saveMeeting(presetValues.meeting, syncBindings: false)
        saveCustomPaste(presetValues.customPaste)
        saveRewriteActivationMode(presetValues.rewriteActivationMode)
        saveTriggerMode(presetValues.triggerMode)
        saveTranscriptionBindings(presetValues.transcriptionBindings)
        saveTranslationBindings(presetValues.translationBindings)
        saveMeetingBindings(presetValues.meetingBindings)
        saveRewriteBindings(presetValues.rewriteBindings)
        saveNoteBindings(presetValues.noteBindings)
    }

    private static func normalizeCustomPasteHotkey(_ hotkey: Hotkey) -> Hotkey {
        guard case .keyboard(let keyCode) = hotkey.input,
              keyCode != modifierOnlyKeyCode
        else { return hotkey }
        return Hotkey(
            input: hotkey.input,
            modifiers: hotkey.modifiers,
            sidedModifiers: []
        )
    }

    private static func legacyDefaultBehavior(defaults: UserDefaults) -> TriggerBehavior {
        TriggerBehavior(loadTriggerMode(defaults: defaults))
    }

    private static func migrateBindingsIfNeeded(
        bindingsKey: String,
        legacyHotkey: Hotkey,
        defaultBehavior: TriggerBehavior,
        defaults: UserDefaults
    ) {
        guard defaults.object(forKey: bindingsKey) == nil else { return }
        saveBindings(
            [.init(hotkey: legacyHotkey, behavior: defaultBehavior)],
            forKey: bindingsKey,
            defaults: defaults
        )
    }

    private static func loadBindings(
        forKey key: String,
        fallbackHotkey: Hotkey,
        defaults: UserDefaults
    ) -> [HotkeyBinding] {
        if let data = defaults.data(forKey: key),
           let decoded = try? JSONDecoder().decode([HotkeyBinding].self, from: data) {
            let sanitized = sanitizedBindings(decoded)
            if !sanitized.isEmpty {
                return sanitized
            }
        }
        let fallback = [HotkeyBinding(hotkey: fallbackHotkey, behavior: legacyDefaultBehavior(defaults: defaults))]
        saveBindings(fallback, forKey: key, defaults: defaults)
        return fallback
    }

    private static func saveBindings(
        _ bindings: [HotkeyBinding],
        forKey key: String,
        defaults: UserDefaults = .standard
    ) {
        let sanitized = sanitizedBindings(bindings)
        guard let data = try? JSONEncoder().encode(sanitized) else { return }
        defaults.set(data, forKey: key)
    }

    private static func sanitizedBindings(_ bindings: [HotkeyBinding]) -> [HotkeyBinding] {
        let sanitized = bindings.map {
            HotkeyBinding(
                id: $0.id,
                hotkey: canonicalHotkey(
                    input: $0.hotkey.input,
                    modifiers: $0.hotkey.modifiers,
                    sidedModifiers: $0.hotkey.sidedModifiers
                ),
                behavior: $0.behavior
            )
        }
        return sanitized.isEmpty
            ? [.init(hotkey: Hotkey(keyCode: defaultKeyCode, modifiers: defaultModifiers, sidedModifiers: []), behavior: .tap)]
            : sanitized
    }

    private static func legacyHotkey(
        inputTypeKey: String,
        keyCodeKey: String,
        mouseButtonKey: String,
        modifiersKey: String,
        sidedModifiersKey: String,
        defaultKeyCode: UInt16,
        defaultModifiers: NSEvent.ModifierFlags,
        defaults: UserDefaults
    ) -> Hotkey {
        let inputTypeRaw = defaults.string(forKey: inputTypeKey)
        let keyCodeValue = defaults.object(forKey: keyCodeKey) as? Int
        let mouseButtonValue = defaults.object(forKey: mouseButtonKey) as? Int
        let modifiersValue = defaults.object(forKey: modifiersKey) as? Int
        let sidedValue = defaults.object(forKey: sidedModifiersKey) as? Int

        let keyCode = keyCodeValue.flatMap { UInt16(exactly: $0) } ?? defaultKeyCode
        let input: Hotkey.Input
        if Hotkey.Input.Kind(rawValue: inputTypeRaw ?? "") == .mouseButton,
           let mouseButtonValue,
           mouseButtonValue >= middleMouseButtonNumber {
            input = .mouseButton(mouseButtonValue)
        } else {
            input = .keyboard(keyCode)
        }
        let modifiersRaw = modifiersValue ?? Int(defaultModifiers.rawValue)
        let modifiers = NSEvent.ModifierFlags(rawValue: UInt(exactly: modifiersRaw) ?? defaultModifiers.rawValue).intersection(.hotkeyRelevant)
        let sidedModifiers = migratedLegacySidedModifiers(
            storedRawValue: sidedValue,
            modifiers: modifiers
        )
        return canonicalHotkey(
            input: input,
            modifiers: modifiers,
            sidedModifiers: sidedModifiers
        )
    }

    private static func migratedLegacySidedModifiers(
        storedRawValue: Int?,
        modifiers: NSEvent.ModifierFlags
    ) -> SidedModifierFlags {
        if let storedRawValue, storedRawValue != 0 {
            return SidedModifierFlags(rawValue: storedRawValue).filtered(by: modifiers)
        }

        var sided: SidedModifierFlags = []
        if modifiers.contains(.shift) { sided.insert(.leftShift) }
        if modifiers.contains(.control) { sided.insert(.leftControl) }
        if modifiers.contains(.option) { sided.insert(.leftOption) }
        if modifiers.contains(.command) { sided.insert(.leftCommand) }
        return sided.filtered(by: modifiers)
    }

    private static func load(
        inputTypeKey: String,
        keyCodeKey: String,
        mouseButtonKey: String,
        modifiersKey: String,
        sidedModifiersKey: String,
        defaultKeyCode: UInt16,
        defaultModifiers: NSEvent.ModifierFlags
    ) -> Hotkey {
        let defaults = UserDefaults.standard
        let inputTypeRaw = defaults.string(forKey: inputTypeKey)
        let keyCodeValue = defaults.object(forKey: keyCodeKey) as? Int
        let mouseButtonValue = defaults.object(forKey: mouseButtonKey) as? Int
        let modifiersValue = defaults.object(forKey: modifiersKey) as? Int
        let sidedValue = defaults.object(forKey: sidedModifiersKey) as? Int ?? 0

        let keyCode = keyCodeValue.flatMap { UInt16(exactly: $0) } ?? defaultKeyCode
        let input: Hotkey.Input
        if Hotkey.Input.Kind(rawValue: inputTypeRaw ?? "") == .mouseButton,
           let mouseButtonValue,
           mouseButtonValue >= middleMouseButtonNumber {
            input = .mouseButton(mouseButtonValue)
        } else {
            input = .keyboard(keyCode)
        }
        let modifiersRaw = modifiersValue ?? Int(defaultModifiers.rawValue)
        let modifiers = NSEvent.ModifierFlags(rawValue: UInt(exactly: modifiersRaw) ?? defaultModifiers.rawValue).intersection(.hotkeyRelevant)
        let sidedModifiers = SidedModifierFlags(rawValue: sidedValue).filtered(by: modifiers)

        return canonicalHotkey(
            input: input,
            modifiers: modifiers,
            sidedModifiers: sidedModifiers
        )
    }

    private static func save(
        _ hotkey: Hotkey,
        inputTypeKey: String,
        keyCodeKey: String,
        mouseButtonKey: String,
        modifiersKey: String,
        sidedModifiersKey: String,
        defaults: UserDefaults
    ) {
        defaults.set(hotkey.input.kind.rawValue, forKey: inputTypeKey)
        switch hotkey.input {
        case .keyboard(let keyCode):
            defaults.set(Int(keyCode), forKey: keyCodeKey)
            defaults.removeObject(forKey: mouseButtonKey)
        case .mouseButton(let buttonNumber):
            defaults.set(buttonNumber, forKey: mouseButtonKey)
        }
        defaults.set(Int(hotkey.modifiers.rawValue), forKey: modifiersKey)
        defaults.set(hotkey.sidedModifiers.filtered(by: hotkey.modifiers).rawValue, forKey: sidedModifiersKey)
    }
}
