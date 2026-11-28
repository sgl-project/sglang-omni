import Darwin

/// note (luojiaxuan): Every timestamp the harness writes is mach_continuous_time() in nanoseconds. It shares
/// mach_absolute_time()'s tick rate but keeps counting while the machine sleeps.
enum MachClock {
    static let timebase: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    static func nanoseconds(ticks: UInt64) -> UInt64 {
        let numer = UInt64(timebase.numer), denom = UInt64(timebase.denom)
        return (ticks / denom) * numer + (ticks % denom) * numer / denom
    }

    static func ticks(nanoseconds ns: UInt64) -> UInt64 {
        let numer = UInt64(timebase.numer), denom = UInt64(timebase.denom)
        return (ns / numer) * denom + (ns % numer) * denom / numer
    }

    /// note (luojiaxuan): mach_continuous_time() in nanoseconds.
    static func now() -> UInt64 {
        nanoseconds(ticks: mach_continuous_time())
    }

    /// note (luojiaxuan): Converts a mach_absolute_time() host time (the clock Core Audio schedules on)
    /// to continuous nanoseconds. The offset between the two clocks only changes
    /// when the machine sleeps; it is read between two continuous-time reads.
    static func continuousNanoseconds(hostTime: UInt64) -> UInt64 {
        let before = mach_continuous_time()
        let absolute = mach_absolute_time()
        let after = mach_continuous_time()
        let midpoint = before + (after - before) / 2
        let offset = midpoint > absolute ? midpoint - absolute : 0
        return nanoseconds(ticks: hostTime + offset)
    }

    /// note (luojiaxuan): Blocks until mach_continuous_time() reaches `deadline` (nanoseconds).
    static func sleep(until deadline: UInt64) {
        while true {
            let now = now()
            if now >= deadline { return }
            let remaining = deadline - now
            var spec = timespec(tv_sec: Int(remaining / 1_000_000_000), tv_nsec: Int(remaining % 1_000_000_000))
            nanosleep(&spec, nil)
        }
    }

    static func seconds(_ s: Double) -> UInt64 {
        UInt64((s * 1e9).rounded())
    }
}
