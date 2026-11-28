import Darwin
import Foundation

/// note (luojiaxuan): Samples the phys_footprint of an app's process tree at a fixed rate.
///
/// The tree is every process whose executable lies inside `<appName>.app/` (the app,
/// its helpers and XPC services, whatever their parent) plus all their descendants by
/// parent pid. It is rebuilt at every tick, so processes that start or exit between
/// ticks are picked up or dropped. Footprints come from proc_pid_rusage, which needs
/// no root for processes of the same user. A read that fails (the process exited
/// during the tick, or belongs to another user) counts as a failure for that tick.
///
/// Output lines (JSON, appended to `outPath`):
///   {"event":"proc","pid":…,"ppid":…,"path":"…","t_mach_continuous_ns":…}   first tick a process is in the tree
///   {"t_mach_continuous_ns":…,"total_phys_footprint":…,"procs":{"<pid>":bytes,…},"failed_pids":[…]}
final class MemorySampler {
    struct Window {
        let peak: UInt64?
        let samples: Int
        let failures: Int
    }

    private struct Tick {
        let t: UInt64
        let total: UInt64
        let failures: Int
    }

    private struct ProcessKey: Hashable {
        let pid: pid_t
        let start: Int
    }

    private let marker: String
    private let interval: Double
    private let out: LineAppender
    private let queue = DispatchQueue(label: "replay.memory-sampler", qos: .userInitiated)
    private var timer: DispatchSourceTimer?
    private let lock = NSLock()
    private var ticks: [Tick] = []
    private var lastTreeSize = -1
    private var paths: [ProcessKey: String] = [:]
    private var announced: Set<ProcessKey> = []

    init(appName: String, hz: Double, outPath: String) throws {
        marker = "/\(appName).app/"
        interval = 1 / hz
        out = try LineAppender(path: outPath)
    }

    func start() {
        let timer = DispatchSource.makeTimerSource(flags: .strict, queue: queue)
        timer.schedule(deadline: .now(), repeating: interval, leeway: .milliseconds(1))
        timer.setEventHandler { [weak self] in self?.tick() }
        self.timer = timer
        timer.resume()
    }

    func stop() {
        queue.sync {
            timer?.cancel()
            timer = nil
        }
    }

    /// note (luojiaxuan): Waits for the first tick and fails when it found no process of the app.
    func requireTarget() throws {
        let giveUp = MachClock.now() + MachClock.seconds(2)
        while MachClock.now() < giveUp {
            lock.lock()
            let size = lastTreeSize
            lock.unlock()
            if size > 0 { return }
            if size == 0 { break }
            usleep(10_000)
        }
        stop()
        throw ReplayError("no running process has an executable inside \(marker); is the app running under that name?")
    }

    /// note (luojiaxuan): The largest summed footprint, the number of ticks and the failed reads in [from, to].
    func window(from: UInt64, to: UInt64) -> Window {
        lock.lock()
        defer { lock.unlock() }
        let inside = ticks.filter { $0.t >= from && $0.t <= to }
        return Window(peak: inside.map(\.total).max(), samples: inside.count, failures: inside.reduce(0) { $0 + $1.failures })
    }

    private func tick() {
        let t = MachClock.now()
        let table = processTable()
        var children: [pid_t: [pid_t]] = [:]
        var queue: [pid_t] = []
        for (pid, entry) in table {
            children[entry.ppid, default: []].append(pid)
            if path(of: ProcessKey(pid: pid, start: entry.start)).contains(marker) { queue.append(pid) }
        }
        var tree: Set<pid_t> = []
        while let pid = queue.popLast() {
            if tree.insert(pid).inserted { queue.append(contentsOf: children[pid] ?? []) }
        }

        var total: UInt64 = 0
        var procs: [String: UInt64] = [:]
        var failed: [Int] = []
        for pid in tree.sorted() {
            let key = ProcessKey(pid: pid, start: table[pid]!.start)
            if announced.insert(key).inserted {
                out.append(["event": "proc", "pid": Int(pid), "ppid": Int(table[pid]!.ppid),
                            "path": path(of: key), "t_mach_continuous_ns": t])
            }
            if let bytes = footprint(pid) {
                total += bytes
                procs[String(pid)] = bytes
            } else {
                failed.append(Int(pid))
            }
        }
        out.append(["t_mach_continuous_ns": t, "total_phys_footprint": total, "procs": procs, "failed_pids": failed])
        lock.lock()
        ticks.append(Tick(t: t, total: total, failures: failed.count))
        lastTreeSize = tree.count
        lock.unlock()
    }

    /// note (luojiaxuan): pid -> (parent pid, start time in microseconds) for every process, from one sysctl.
    private func processTable() -> [pid_t: (ppid: pid_t, start: Int)] {
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_ALL, 0]
        var size = 0
        guard sysctl(&mib, 3, nil, &size, nil, 0) == 0 else { return [:] }
        // note (luojiaxuan): The table can grow between the two calls.
        size += 64 * MemoryLayout<kinfo_proc>.stride
        var procs = [kinfo_proc](repeating: kinfo_proc(), count: size / MemoryLayout<kinfo_proc>.stride)
        guard sysctl(&mib, 3, &procs, &size, nil, 0) == 0 else { return [:] }
        var table: [pid_t: (ppid: pid_t, start: Int)] = [:]
        for proc in procs.prefix(size / MemoryLayout<kinfo_proc>.stride) {
            let start = proc.kp_proc.p_un.__p_starttime
            table[proc.kp_proc.p_pid] = (proc.kp_eproc.e_ppid, start.tv_sec * 1_000_000 + Int(start.tv_usec))
        }
        return table
    }

    /// note (luojiaxuan): The executable path, cached per process (a reused pid has a new start time).
    private func path(of key: ProcessKey) -> String {
        if let cached = paths[key] { return cached }
        var buffer = [CChar](repeating: 0, count: 4 * Int(MAXPATHLEN))
        let length = proc_pidpath(key.pid, &buffer, UInt32(buffer.count))
        let path = length > 0 ? String(cString: buffer) : ""
        paths[key] = path
        return path
    }

    private func footprint(_ pid: pid_t) -> UInt64? {
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(pid, RUSAGE_INFO_V4, $0) }
        }
        return status == 0 ? info.ri_phys_footprint : nil
    }
}
