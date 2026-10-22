/// Splits typed letters into pinyin syllables. Fewest syllables wins ("xian" is
/// one syllable; "xi'an" forces two). Ties go to the longer first syllable
/// ("wanan" → wan-an). An initial stands for a whole syllable (简拼: "nh" is
/// n-h) where full syllables don't fit. Letters no syllable or initial can
/// start (i, u, v) stay raw, with everything after them.
public struct PinyinSegmenter: Sendable {
    public struct Split: Equatable, Sendable {
        public var syllables: [String]
        public var rest: String
    }

    /// Toneless syllables in the rime-wanxiang data (ü typed as v), without
    /// the vowelless interjections m, n and ng. 415 entries.
    public static let syllables: Set<String> = Set("""
        a ai an ang ao ba bai ban bang bao bei ben beng bi bian biang biao bie bin bing bo bu ca cai can cang cao ce
        cei cen ceng cha chai chan chang chao che chen cheng chi chong chou chu chua chuai chuan chuang chui chun
        chuo ci cong cou cu cuan cui cun cuo da dai dan dang dao de dei den deng di dia dian diao die ding diu dong
        dou du duan dui dun duo e ei en eng er fa fan fang fei fen feng fiao fo fou fu ga gai gan gang gao ge gei gen
        geng gong gou gu gua guai guan guang gui gun guo ha hai han hang hao he hei hen heng hong hou hu hua huai
        huan huang hui hun huo ji jia jian jiang jiao jie jin jing jiong jiu ju juan jue jun ka kai kan kang kao ke
        kei ken keng kong kou ku kua kuai kuan kuang kui kun kuo la lai lan lang lao le lei leng li lia lian liang
        liao lie lin ling liu lo long lou lu luan lun luo lv lve ma mai man mang mao me mei men meng mi mian miao mie
        min ming miu mo mou mu na nai nan nang nao ne nei nen neng ni nian niang niao nie nin ning niu nong nou nu
        nuan nun nuo nv nve o ou pa pai pan pang pao pei pen peng pi pian piao pie pin ping po pou pu qi qia qian
        qiang qiao qie qin qing qiong qiu qu quan que qun ran rang rao re ren reng ri rong rou ru rua ruan rui run
        ruo sa sai san sang sao se sen seng sha shai shan shang shao she shei shen sheng shi shou shu shua shuai
        shuan shuang shui shun shuo si song sou su suan sui sun suo ta tai tan tang tao te tei teng ti tian tiao tie
        ting tong tou tu tuan tui tun tuo wa wai wan wang wei wen weng wo wu xi xia xian xiang xiao xie xin xing
        xiong xiu xu xuan xue xun ya yan yang yao ye yi yin ying yo yong you yu yuan yue yun za zai zan zang zao ze
        zei zen zeng zha zhai zhan zhang zhao zhe zhei zhen zheng zhi zhong zhou zhu zhua zhuai zhuan zhuang zhui
        zhun zhuo zi zong zou zu zuan zui zun zuo
        """.split(whereSeparator: \.isWhitespace).map(String.init))
    /// Initials that stand for a whole syllable; zh, ch and sh count as one.
    public static let initials: Set<String> = Set("b p m f d t n l g k h j q x zh ch sh r z c s y w".split(separator: " ").map(String.init))

    /// Every syllable with its other z/zh, c/ch, s/sh spelling ("zuang" for zhuang).
    private static let fuzzySyllables = syllables.union(syllables.map { flipped($0) })

    private let pieces: Set<String>
    private let longest: Int

    /// `fuzzy`: z/zh, c/ch and s/sh spellings both split as syllables.
    public init(fuzzy: Bool = false) {
        pieces = (fuzzy ? Self.fuzzySyllables : Self.syllables).union(Self.initials)
        longest = pieces.map(\.count).max() ?? 6
    }

    /// Whether a typed piece (a syllable or an initial) can stand for a dictionary syllable.
    public static func matches(_ typed: String, _ syllable: String, fuzzy: Bool) -> Bool {
        if initials.contains(typed) {
            let initial = initial(of: syllable)
            return fuzzy ? flat(initial) == flat(typed) : initial == typed
        }
        if typed == syllable { return true }
        guard fuzzy else { return false }
        return flipped(typed) == syllable
    }

    /// zh/ch/sh, or the first letter when it is an initial; "" for a, e, o syllables.
    static func initial(of syllable: String) -> String {
        for double in ["zh", "ch", "sh"] where syllable.hasPrefix(double) { return double }
        let first = String(syllable.prefix(1))
        return initials.contains(first) ? first : ""
    }

    private static func flat(_ initial: String) -> String {
        initial.count == 2 ? String(initial.prefix(1)) : initial
    }

    /// The other spelling of a z/zh, c/ch, s/sh syllable; others unchanged.
    static func flipped(_ syllable: String) -> String {
        switch initial(of: syllable) {
        case "zh", "ch", "sh": String(syllable.prefix(1)) + syllable.dropFirst(2)
        case "z", "c", "s": String(syllable.prefix(1)) + "h" + syllable.dropFirst(1)
        default: syllable
        }
    }

    public static let maximumSplits = 8

    public func split(_ input: String) -> Split { splits(input)[0] }

    /// Every fewest-syllable split of the letters that fit, best guess first
    /// (longer first syllables), at most `maximumSplits`. The rest is the same
    /// for all of them.
    public func splits(_ input: String) -> [Split] {
        let letters = Array(input)
        var choices: [[[String]]] = []  // per apostrophe-separated run
        var start = 0
        while start < letters.count {
            if letters[start] == "'" { start += 1; continue }
            var end = start
            while end < letters.count, letters[end] != "'" { end += 1 }
            let (runSplits, covered) = cover(Array(letters[start..<end]))
            if !runSplits.isEmpty { choices.append(runSplits) }
            if covered < end - start {
                return Self.combine(choices, rest: String(letters[(start + covered)...]))
            }
            start = end
        }
        // Keep a trailing apostrophe so the next letter starts a new syllable.
        return Self.combine(choices, rest: input.hasSuffix("'") && !choices.isEmpty ? "'" : "")
    }

    private static func combine(_ choices: [[[String]]], rest: String) -> [Split] {
        var result: [[String]] = [[]]
        for options in choices {
            result = Array(result.lazy.flatMap { head in options.map { head + $0 } }.prefix(maximumSplits))
        }
        return result.map { Split(syllables: $0, rest: rest) }
    }

    /// The longest prefix of `text` that splits into syllables, with all its
    /// fewest-syllable splits.
    private func cover(_ text: [Character]) -> ([[String]], Int) {
        for end in stride(from: text.count, through: 1, by: -1) {
            let found = solve(text, end: end)
            if !found.isEmpty { return (found, end) }
        }
        return ([], 0)
    }

    /// Fewest pieces first, then fewest initials: a piece costs 10, an initial 11.
    private func solve(_ text: [Character], end: Int) -> [[String]] {
        // best[i]: lowest cost covering text[i..<end]; nil = impossible.
        var best = [Int?](repeating: nil, count: end + 1)
        best[end] = 0
        func piece(_ i: Int, _ length: Int) -> String? {
            let piece = String(text[i..<i + length])
            return pieces.contains(piece) ? piece : nil
        }
        func cost(_ piece: String) -> Int { Self.initials.contains(piece) ? 11 : 10 }
        for i in stride(from: end - 1, through: 0, by: -1) {
            for length in 1...min(longest, end - i) {
                guard let piece = piece(i, length), let after = best[i + length] else { continue }
                if best[i].map({ after + cost(piece) < $0 }) ?? true { best[i] = after + cost(piece) }
            }
        }
        guard best[0] != nil else { return [] }
        var found: [[String]] = []
        func walk(_ i: Int, _ parts: [String]) {
            guard found.count < Self.maximumSplits else { return }
            if i == end { found.append(parts); return }
            for length in stride(from: min(longest, end - i), through: 1, by: -1) {
                guard let piece = piece(i, length), let after = best[i + length], after + cost(piece) == best[i]! else { continue }
                walk(i + length, parts + [piece])
            }
        }
        walk(0, [])
        return found
    }
}
