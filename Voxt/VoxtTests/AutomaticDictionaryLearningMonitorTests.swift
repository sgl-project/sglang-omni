// AutomaticDictionaryLearningMonitorTests.swift
// Provides Automatic Dictionary Learning Monitor Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class AutomaticDictionaryLearningMonitorTests: XCTestCase {
    func testAutoKeyPressSkipsObservationScheduling() {
        let decision = AutomaticDictionaryLearningMonitor.observationScheduleDecision(
            didInject: true,
            isTranscriptionOutput: true,
            isFeatureEnabled: true,
            insertedText: "Send this message",
            didTriggerAutoKeyPress: true
        )

        XCTAssertEqual(decision, .skipAutoKeyPress)
    }

    func testTranscriptionWithoutAutoKeyPressSchedulesObservation() {
        let decision = AutomaticDictionaryLearningMonitor.observationScheduleDecision(
            didInject: true,
            isTranscriptionOutput: true,
            isFeatureEnabled: true,
            insertedText: "Keep editing this draft",
            didTriggerAutoKeyPress: false
        )

        XCTAssertEqual(decision, .schedule)
    }

    func testBuildsLearningRequestForInPlaceCorrection() {
        let outcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: "anthropic ai",
            baselineText: "Please ship anthropic ai today.",
            finalText: "Please ship Anthropic today."
        )

        guard case .ready(let request) = outcome else {
            return XCTFail("Expected ready outcome, got \(outcome)")
        }

        XCTAssertEqual(request.insertedText, "anthropic ai")
        XCTAssertEqual(request.baselineChangedFragment, "anthropic ai")
        XCTAssertEqual(request.finalChangedFragment, "Anthropic")
        XCTAssertLessThanOrEqual(request.editRatio, AutomaticDictionaryLearningMonitor.maximumEditRatio)
    }

    func testBuildsLearningRequestForEqualLengthReplacement() {
        let outcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: "Waxed",
            baselineText: "Our app is named Waxed.",
            finalText: "Our app is named Voxt."
        )

        guard case .ready(let request) = outcome else {
            return XCTFail("Expected ready outcome, got \(outcome)")
        }

        XCTAssertEqual(request.baselineChangedFragment, "Waxed.")
        XCTAssertEqual(request.finalChangedFragment, "Voxt.")
        XCTAssertLessThanOrEqual(request.editRatio, AutomaticDictionaryLearningMonitor.maximumEditRatio)
    }

    func testBuildsLearningRequestForAdjacentEnglishPhraseCorrection() {
        let outcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: "你好，您来帮我看一下我们新的公众号里面的 code code 有什么文章。",
            baselineText: "你好，您来帮我看一下我们新的公众号里面的 code code 有什么文章。",
            finalText: "你好，您来帮我看一下我们新的公众号里面的 claude code 有什么文章。"
        )

        guard case .ready(let request) = outcome else {
            return XCTFail("Expected ready outcome, got \(outcome)")
        }

        XCTAssertEqual(request.baselineChangedFragment, "code")
        XCTAssertEqual(request.finalChangedFragment, "claude")
        XCTAssertLessThanOrEqual(request.editRatio, AutomaticDictionaryLearningMonitor.maximumEditRatio)
    }

    func testBuildsLearningRequestForMultiClauseChineseCorrectionWithoutMergingWholeSentence() {
        let outcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: "看一下我们投坑中有没有新的词源了。这个新的词源也需要接飞。",
            baselineText: "看一下我们投坑中有没有新的词源了。这个新的词源也需要接飞。",
            finalText: "看一下我们投坑中有没有新的词元了。这个新的词元也需要接入。"
        )

        guard case .ready(let request) = outcome else {
            return XCTFail("Expected ready outcome, got \(outcome)")
        }

        XCTAssertEqual(request.baselineChangedFragment, "看一下我们投坑中有没有新的词源了。这个新的词源也需要接飞")
        XCTAssertEqual(request.finalChangedFragment, "看一下我们投坑中有没有新的词元了。这个新的词元也需要接入")
        XCTAssertLessThanOrEqual(request.editRatio, AutomaticDictionaryLearningMonitor.maximumEditRatio)
    }

    func testBuildsLearningRequestForDomainCorrectionParagraphs() {
        struct Case {
            let name: String
            let insertedText: String
            let finalText: String
            let expectedBaselineFragment: String
            let expectedFinalFragment: String
        }

        let cases: [Case] = [
            Case(
                name: "programmer refactor note",
                insertedText: """
                模块三用户鉴权终于重构完成了。以前那套session逻辑并发高了就崩溃，这次换成J W T加瑞的斯，性能提升了五倍还多。踩了个坑，刷新token的时候忘了校验黑名单，差点留了个安全漏洞。明天再加个中间件，顺便把日志的级别调成info。产品经理说下个版本要加人脸识别登录，评估了下，估计再加一个活体检测服务就行了。预算顶翻倍了。
                """,
                finalText: """
                模块3用户鉴权终于重构完了。以前那套Session逻辑并发高了就崩溃，这次换成JWT + Redis，性能提升5倍不止。踩了个坑：刷新Token的时候忘了校验黑名单，差点留个安全漏洞。明天加个中间件，顺便把日志级别调成INFO。产品经理说下个版本要加人脸登录，评估了下，估计得再搭一个活体检测服务，预算得翻倍了。
                """,
                expectedBaselineFragment: "J W T加瑞的斯",
                expectedFinalFragment: "JWT + Redis"
            ),
            Case(
                name: "nurse handoff note",
                insertedText: """
                四十三床张爷爷，COPD 急性发作，凌晨两点血氧掉到百分之八十八，无创通气后恢复至百分之九十四，血氧稳定。乙中乙左加甲泼尼龙四十毫克。本人人左，左胸闷，建议日间查心肌酶。十五床李女士，剖腹产术后第二天，子宫收缩好，已拔尿管，自洁小便顺畅，注意六床新收发热患儿，流感抗原阳性，已隔离，请日间医生重点关注。
                """,
                finalText: """
                43床张爷爷，COPD急性发作，凌晨2点血氧掉到88%，无创通气后恢复至94%，血压稳定。已遵医嘱加甲泼尼龙40mg。本人仍主诉胸闷，建议日间查心肌酶。15床李女士，剖腹产术后第二天，子宫收缩好，已拔尿管，自解小便顺畅。注意：6床新收发热患儿，流感抗原阳性，已隔离。请日间医生重点关注。
                """,
                expectedBaselineFragment: "乙中乙左",
                expectedFinalFragment: "已遵医嘱"
            ),
            Case(
                name: "chef menu development note",
                insertedText: """
                试了第三把黑松露炒饭：第一把松露油放太多，腻；第二把加了性保锅里，增加口感，但米饭不够干爽。今天改用隔夜的泰国香米，煸干水分之后下黑松露酱，最后撒一点盐渍花提味。主厨尝了说，对了，但建议把配的温泉蛋换成溏心煎蛋，卖相更好。成本核算大概十二块，菜单定价六十八，毛利还行。
                """,
                finalText: """
                试了第三版“黑松露炒饭”。第一版松露油放太多，腻；第二版加了杏鲍菇粒增加口感，但米饭不够干爽。今天改用隔夜泰国香米，煸干水分后下黑松露酱，最后撒一点点盐之花提味。主厨尝了说“对了”，但建议把配的温泉蛋换成溏心煎蛋，卖相更好。成本核算大概12块，菜单定价68，毛利还行。
                """,
                expectedBaselineFragment: "性保锅里",
                expectedFinalFragment: "杏鲍菇粒"
            ),
            Case(
                name: "lawyer case strategy note",
                insertedText: """
                关于王某诉某科技公司敬业限制纠纷一案，关键点在于公司所主张的核心算法工程师身份是否有足够证据。目前我方掌握王某入职第三个月即调岗至非技术部门，且从未接触代码库的邮件记录。准备申请法院调取其社保缴纳、职位记录。另外，敬业补偿金一直未足额支付，可能成为合同解除的突破口。
                """,
                finalText: """
                关于王某诉某科技公司竞业限制纠纷一案，关键点在于：公司所主张的“核心算法工程师”身份，是否有足够证据。目前我方掌握王某入职第三个月即调岗至非技术部门，且从未接触代码库的邮件记录。准备申请法院调取其社保缴纳职位记录。另外，竞业补偿金一直未足额支付，可能成为合同解除的突破口。
                """,
                expectedBaselineFragment: "敬业限制",
                expectedFinalFragment: "竞业限制"
            ),
            Case(
                name: "teacher weekly report",
                insertedText: """
                本周完成期中考试大分统计，班级平均分比年级低 2.3 分，主要是数学拖后腿，已联系数学老师增加周日下午自习辅导。重点关注小陈同学连续三天迟到，家长反馈晚上打游戏到凌晨，已约谈。下周班会主题定为时间管理，准备请上一届学长来分享。另外，教室投影仪灯泡发红，报修单已提交。
                """,
                finalText: """
                本周完成：期中考试分析，班级平均分比年级低2.3分，主因是数学拖后腿。已联系数学老师增加周日下午自习辅导。重点关注：小陈同学连续三天迟到，家长反馈晚上打游戏到凌晨，已约谈。下周班会主题定为“时间管理”，准备请上一届学长来分享。另外，教室投影仪灯泡发红，报修单已提交。
                """,
                expectedBaselineFragment: "期中考试大分统计",
                expectedFinalFragment: "期中考试分析"
            ),
            Case(
                name: "architect site inspection note",
                insertedText: """
                商业综合体三层中庭钢构安装，现场发现次梁连接板开孔偏差五毫米，已要求工人停止焊接，与钢构厂沟通，同意补送一批连接板，预计明天下午到。幕墙预埋件位置符合，西南角缺三个，责令土建班组限期补埋，下午协调精装与机电：空调风管与吊顶龙骨冲突，建议风管改走梁窝，代价最小。
                """,
                finalText: """
                商业综合体三层中庭钢构安装，现场发现次梁连接板开孔偏差5mm，已要求工人停止焊接。与钢构厂沟通，同意补送一批连接板，预计明天下午到。幕墙预埋件位置复核，西南角缺三个，责令土建班组限期补埋。下午协调精装与机电：空调风管与吊顶龙骨冲突，建议风管改走梁窝，代价最小。
                """,
                expectedBaselineFragment: "位置符合",
                expectedFinalFragment: "位置复核"
            ),
            Case(
                name: "ecommerce daily report",
                insertedText: """
                访客数八千七百，同比降百分之五。佐烟精品双十二返场活动，转化率百分之二点一，低于目标。爆款 A 的广告花费占比升到百分之二十一，考虑明天下调出价。新增差评一条，买家投诉包装破损，客服已补偿十元券。建议包国内增加破损包赔卡片。另，观察两款新品加购率不错，可尝试今日开一个百分之十折扣的限时秒杀。
                """,
                finalText: """
                访客数8700，同比降5%，主因竞品“双12返场”活动。转化率2.1%，低于目标。爆款A的广告花费占比升到21%，考虑明天下调出价。新增差评一条：买家投诉包装破损，客服已补偿10元券。建议：包裹内增加“破损包赔”卡片。另，观察两款新品加购率不错，可尝试今日开一个10%折扣的限时秒杀。
                """,
                expectedBaselineFragment: "包国内",
                expectedFinalFragment: "包裹内"
            ),
            Case(
                name: "therapist case note",
                insertedText: """
                男方者女，二十八岁，左述职场焦虑，伴失眠两个月。投射测验显示高自我与要求，与低自我效能感并存。本次重点探索其必须完美的核心信念，不止行为实验。故意在工作群发一条带错别字的消息，观察追化结果是否发生。下周反馈，需要注意躯体化症状，手抖、心慌，建议排除甲亢，脂肪关系初步建立良好。
                """,
                finalText: """
                来访者，女，28岁，主诉职场焦虑伴失眠两个月。投射测验显示高自我要求与低自我效能感并存。本次重点探索其“必须完美”的核心信念，布置行为实验：故意在工作群发一条带错别字的消息，观察最坏结果是否发生。下周反馈。需要注意躯体化症状（手抖、心慌），建议排除甲亢。咨访关系初步建立良好。
                """,
                expectedBaselineFragment: "脂肪关系",
                expectedFinalFragment: "咨访关系"
            ),
            Case(
                name: "firefighter rescue report",
                insertedText: """
                十一月二十五日 14:32 接警，成都物流园一辆货车自燃，载有纸箱和少量油气，出警两车十二人。到达时火势呈猛烈燃烧阶段，立即出两支水枪，一支灭火，一支冷却油箱。13:10 明火扑灭，持续降温二十分钟，无人员伤亡，过火面积约八平方米。原因初步判断为电气线路老化，建议对园区所有货车进行电路排查。
                """,
                finalText: """
                11月25日14:32接警，城东物流园一辆货车自燃，载有纸箱和少量油漆。出警两车12人，到达时火势呈猛烈燃烧阶段。立即出两支水枪，一支灭火一支冷却油箱。15:10明火扑灭，持续降温20分钟。无人员伤亡，过火面积约8平方米。原因初步判断为电气线路老化。建议对园区所有货车进行电路排查。
                """,
                expectedBaselineFragment: "成都物流园",
                expectedFinalFragment: "城东物流园"
            ),
            Case(
                name: "screenwriter outline note",
                insertedText: """
                第一稿被毙了，说悬疑线太复杂，观众看不懂。第二稿决定砍掉一条副线，把凶手从双人改成单人，男主角的职业从记者改成片警，更接地气。加了一场雨夜追车的动作戏，预算可能会超，但节奏好。结尾反转保留，但提前到第三幕开头，留出十五分钟给情感宣泄。下周交分场大纲，争取过会。
                """,
                finalText: """
                第一稿被毙了，说“悬疑线太复杂，观众看不懂”。第二稿决定砍掉一条副线，把凶手从双人改成单人。男主角的职业从记者改成片警，更接地气。加了一场雨夜追车的动作戏，预算可能会超，但节奏好。结尾反转保留，但提前到第三幕开头，留出15分钟给情感宣泄。下周交分场大纲，争取过会。
                """,
                expectedBaselineFragment: "十五分钟",
                expectedFinalFragment: "15分钟"
            )
        ]

        for item in cases {
            let finalText = item.insertedText.replacingOccurrences(
                of: item.expectedBaselineFragment,
                with: item.expectedFinalFragment
            )
            XCTAssertNotEqual(finalText, item.insertedText, "Fixture target missing for \(item.name)")
            assertLearningDiff(
                insertedText: item.insertedText,
                finalText: finalText,
                expectedBaselineFragment: item.expectedBaselineFragment,
                expectedFinalFragment: item.expectedFinalFragment,
                message: item.name
            )
        }
    }

    func testDirectCandidateTermsFallbackReturnsFinalCorrectedToolName() {
        let request = AutomaticDictionaryLearningRequest(
            insertedText: "帮我看一下我们的 WeChat 里面有没有 Cloud Code 新发的消息。",
            baselineContext: "帮我看一下我们的 WeChat 里面有没有 Cloud Code 新发的消息。",
            finalContext: "帮我看一下我们的 WeChat 里面有没有 Claude Code 新发的消息。",
            baselineChangedFragment: "Cloud Code",
            finalChangedFragment: "Claude Code",
            editRatio: 0.12
        )

        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.directCandidateTerms(
                for: request,
                existingTerms: ["Voxt"]
            ),
            ["Claude Code"]
        )
    }

    func testSkipsDeletionOnlyChangeWhenBaselineWrapsInsertedTextAcrossLines() {
        let outcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: "React JS 和 Next JS",
            baselineText: "我们使用了 React JS\n和 Next JS 来实现整个 APP 的链路。",
            finalText: "我们使用了 React 和 Next JS 来实现整个 APP 的链路。"
        )

        assertSkipped(outcome, contains: "changed fragment has no meaningful terms")
    }

    func testBuildsLearningRequestFromTerminalLineWhenFinalSnapshotContainsCommandOutput() {
        let outcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: "你帮我看一下我们的 Go Host 能不能识别我们现在 APP 中的内容呀？",
            baselineText: """
            ~/x/doit/voxt-service
            > 你帮我看一下我们的 Go Host 能不能识别我们现在 APP 中的内容呀？
            """,
            finalText: """
            > 你帮我看一下我们的 Ghostty 能不能识别我们现在 APP 中的内容呀？
            zsh: command not found: 你帮我看一下我们的

            ~/x/doit/voxt-service
            """
        )

        guard case .ready(let request) = outcome else {
            return XCTFail("Expected ready outcome, got \(outcome)")
        }

        XCTAssertEqual(request.baselineChangedFragment, "Go Host")
        XCTAssertEqual(request.finalChangedFragment, "Ghostty")
        XCTAssertLessThanOrEqual(request.editRatio, AutomaticDictionaryLearningMonitor.maximumEditRatio)
    }

    func testObservationScopedTextPrefersEchoedCommandLineAfterPromptClears() {
        let scopedText = AutomaticDictionaryLearningMonitor.observationScopedText(
            insertedText: "看一下我们投坑中有没有新的词源了。这个新的词源是也需要接飞的。",
            baselineText: """
            ~/x/doit/voxt-service  main !20 ?9
            > 看一下我们投坑中有没有新的词源了。这个新的词源是也需要接飞的。
            """,
            currentText: """
            > 看一下我们 token 中有没有新的词元了。这个新的词元也需要计费。
            zsh: command not found: 看一下我们

            ~/x/doit/voxt-service  main !20 ?9
            >
            """
        )

        XCTAssertEqual(
            scopedText,
            "看一下我们 token 中有没有新的词元了。这个新的词元也需要计费。"
        )
    }

    func testObservationSettlesAfterEchoedCommandContainsCompletedReplacement() {
        let baselineText = "看一下我们投坑中有没有新的词源了。这个新的词源是也需要接飞的。"
        let echoedFinalText = "看一下我们 token 中有没有新的词元了。这个新的词元也需要计费。"

        XCTAssertFalse(
            AutomaticDictionaryLearningMonitor.shouldContinueObservingForPotentialReplacement(
                baselineText: baselineText,
                currentFinalText: echoedFinalText
            )
        )
    }

    func testBuildsLearningRequestWithExpandedTokenBoundaryFragments() {
        let outcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: "你帮我识别一下 WeChat 中的输入文本，我们是支持的吗？还有我们要支持一下 SG 狼魔鬼穷的文本查询。",
            baselineText: "你帮我识别一下 WeChat 中的输入文本，我们是支持的吗？还有我们要支持一下 SG 狼魔鬼穷的文本查询。",
            finalText: "你帮我识别一下 WeChat 中的输入文本，我们是支持的吗？还有我们要支持一下 SGLang 魔鬼群的文本查询。"
        )

        guard case .ready(let request) = outcome else {
            return XCTFail("Expected ready outcome, got \(outcome)")
        }

        XCTAssertTrue(request.baselineChangedFragment.contains("狼"))
        XCTAssertTrue(request.finalChangedFragment.contains("Lang"))
        XCTAssertLessThanOrEqual(request.editRatio, AutomaticDictionaryLearningMonitor.maximumEditRatio)
    }

    func testBuildsLearningRequestByComparingInsertedTextAgainstFinalScopedLine() {
        let outcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: "你好，你帮我们看一下 WeChat 中的 SG 骆魔鬼群，他们用户在说什么？",
            baselineText: """
            > 你帮我看一下我们的 Ghostty 能不能识别我们现在 APP 中的内容呀？
            zsh: command not found: 你帮我看一下我们的

             ~/x/doit/voxt-service  main !20 ?9
            > 你好，你帮我们看一下 WeChat 中的 SG 骆魔鬼群，他们用户在说什么？
            """,
            finalText: """
            > 你帮我看一下我们的 Ghostty 能不能识别我们现在 APP 中的内容呀？
            zsh: command not found: 你帮我看一下我们的
            > 你好，你帮我们看一下 WeChat 中的 SGLang魔鬼群，他们用户在说什么？
            zsh: command not found: 你好，你帮我们看一下

             ~/x/doit/voxt-service  main !20 ?9
            >
            """
        )

        guard case .ready(let request) = outcome else {
            return XCTFail("Expected ready outcome, got \(outcome)")
        }

        XCTAssertEqual(
            request.baselineContext,
            "你好，你帮我们看一下 WeChat 中的 SG 骆魔鬼群，他们用户在说什么？"
        )
        XCTAssertEqual(
            request.finalContext,
            "你好，你帮我们看一下 WeChat 中的 SGLang魔鬼群，他们用户在说什么？"
        )
        XCTAssertEqual(request.baselineChangedFragment, "骆")
        XCTAssertEqual(request.finalChangedFragment, "SGLang魔鬼群")
        XCTAssertLessThanOrEqual(request.editRatio, AutomaticDictionaryLearningMonitor.maximumEditRatio)
    }

    func testContinuesObservingWhenLatestEditLooksLikeDeletionOnly() {
        XCTAssertTrue(
            AutomaticDictionaryLearningMonitor.shouldContinueObservingForPotentialReplacement(
                baselineText: "我们配合 Go Hoste 来实现 Terminal CLI 的输入。",
                currentFinalText: "我们配合  来实现 Terminal CLI 的输入。"
            )
        )
    }

    func testDoesNotContinueObservingWhenReplacementTextAlreadyExists() {
        XCTAssertFalse(
            AutomaticDictionaryLearningMonitor.shouldContinueObservingForPotentialReplacement(
                baselineText: "我们配合 Go Hoste 来实现 Terminal CLI 的输入。",
                currentFinalText: "我们配合 Ghostty 来实现 Terminal CLI 的输入。"
            )
        )
    }

    func testObservationStopsAfterConsecutiveMissingSnapshotsBeforeAnyEdit() {
        var state = AutomaticDictionaryLearningObservationState(
            baselineText: "baseline"
        )

        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state),
            .continueObserving
        )
        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state),
            .continueObserving
        )
        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state),
            .stopWithoutAnalysis
        )
    }

    func testObservationSettlesAfterConsecutiveMissingSnapshotsOnceEditIsIdle() {
        var state = AutomaticDictionaryLearningObservationState(
            baselineText: "我们配合 Go Hoste 来实现 Terminal CLI 的输入。"
        )
        state.latestText = "我们配合 Ghostty 来实现 Terminal CLI 的输入。"
        state.didObserveChange = true
        state.lastChangeElapsedSeconds = AutomaticDictionaryLearningMonitor.idleSettleSeconds + 0.1

        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state),
            .continueObserving
        )
        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state),
            .continueObserving
        )
        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state),
            .settleForAnalysis(finalText: "我们配合 Ghostty 来实现 Terminal CLI 的输入。")
        )
    }

    func testObservationDoesNotSettleAfterMissingSnapshotsBeforeIdleThreshold() {
        var state = AutomaticDictionaryLearningObservationState(
            baselineText: "我们配合 Go Hoste 来实现 Terminal CLI 的输入。"
        )
        state.latestText = "我们配合 Ghostty 来实现 Terminal CLI 的输入。"
        state.didObserveChange = true
        state.lastChangeElapsedSeconds = AutomaticDictionaryLearningMonitor.idleSettleSeconds - 0.1

        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state),
            .continueObserving
        )
        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state),
            .continueObserving
        )
        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state),
            .continueObserving
        )
    }

    func testObservationDoesNotSettleAfterMissingSnapshotsWhenReplacementStillIncomplete() {
        var state = AutomaticDictionaryLearningObservationState(
            baselineText: "你好，你帮我看一下我们的微信中有没有新的消息，特别是 Cloud Code 和我们的 Go Host。我们为了识别 Terminal CLI，做了一些优化。"
        )
        state.latestText = "你好，你帮我看一下我们的微信中有没有新的消息，特别是 Claude Code 和我们的 。我们为了识别 Terminal CLI，做了一些优化。"
        state.didObserveChange = true
        state.lastChangeElapsedSeconds = AutomaticDictionaryLearningMonitor.idleSettleSeconds + 0.1

        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state),
            .continueObserving
        )
        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state),
            .continueObserving
        )
        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state),
            .continueObserving
        )
    }

    func testObservationContinuesWhenIdleSnapshotStillLooksLikeDeletionOnly() {
        var state = AutomaticDictionaryLearningObservationState(
            baselineText: "我们配合 Go Hoste 来实现 Terminal CLI 的输入。"
        )
        state.latestText = "我们配合  来实现 Terminal CLI 的输入。"
        state.didObserveChange = true

        let decision = AutomaticDictionaryLearningMonitor.observeSnapshot(
            text: "我们配合  来实现 Terminal CLI 的输入。",
            elapsedSinceLastChange: AutomaticDictionaryLearningMonitor.idleSettleSeconds + 0.1,
            state: &state
        )

        XCTAssertEqual(decision, .continueObserving)
    }

    func testObservationContinuesWhenIdleSnapshotStillContainsUnfinishedDeletionGroup() {
        var state = AutomaticDictionaryLearningObservationState(
            baselineText: "你好，你帮我看一下我们的微信中有没有新的消息，特别是 Cloud Code 和我们的 Go Host。我们为了识别 Terminal CLI，做了一些优化。"
        )
        state.latestText = "你好，你帮我看一下我们的微信中有没有新的消息，特别是 Claude Code 和我们的 。我们为了识别 Terminal CLI，做了一些优化。"
        state.didObserveChange = true

        let decision = AutomaticDictionaryLearningMonitor.observeSnapshot(
            text: state.latestText,
            elapsedSinceLastChange: AutomaticDictionaryLearningMonitor.idleSettleSeconds + 0.1,
            state: &state
        )

        XCTAssertEqual(decision, .continueObserving)
    }

    func testObservationSettlesWhenIdleSnapshotContainsCompletedReplacement() {
        var state = AutomaticDictionaryLearningObservationState(
            baselineText: "我们配合 Go Hoste 来实现 Terminal CLI 的输入。"
        )
        state.latestText = "我们配合 Ghostty 来实现 Terminal CLI 的输入。"
        state.didObserveChange = true

        let decision = AutomaticDictionaryLearningMonitor.observeSnapshot(
            text: "我们配合 Ghostty 来实现 Terminal CLI 的输入。",
            elapsedSinceLastChange: AutomaticDictionaryLearningMonitor.idleSettleSeconds + 0.1,
            state: &state
        )

        XCTAssertEqual(
            decision,
            .settleForAnalysis(finalText: "我们配合 Ghostty 来实现 Terminal CLI 的输入。")
        )
    }

    func testStableFocusedEditDoesNotFinalizeObservationImmediately() {
        XCTAssertFalse(
            AutomaticDictionaryLearningMonitor.shouldFinalizeWhileFocused(
                decision: .settleForAnalysis(finalText: "Claude Code")
            )
        )
    }

    func testObservationResetsMissingCounterWhenFocusedSnapshotReturns() {
        var state = AutomaticDictionaryLearningObservationState(
            baselineText: "baseline"
        )
        _ = AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state)
        _ = AutomaticDictionaryLearningMonitor.observeMissingSnapshot(state: &state)

        let decision = AutomaticDictionaryLearningMonitor.observeSnapshot(
            text: "baseline",
            elapsedSinceLastChange: nil,
            state: &state
        )

        XCTAssertEqual(decision, .continueObserving)
        XCTAssertEqual(state.consecutiveMissingSnapshots, 0)
        XCTAssertFalse(state.didObserveChange)
    }

    func testObservationSettlesAfterDeletionIntermediateWhenReplacementIsCompleted() {
        var state = AutomaticDictionaryLearningObservationState(
            baselineText: "我们配合 Go Hoste 来实现 Terminal CLI 的输入。"
        )
        state.latestText = "我们配合  来实现 Terminal CLI 的输入。"
        state.didObserveChange = true

        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeSnapshot(
                text: "我们配合 Ghostty 来实现 Terminal CLI 的输入。",
                elapsedSinceLastChange: 0.2,
                state: &state
            ),
            .continueObserving
        )

        XCTAssertEqual(
            AutomaticDictionaryLearningMonitor.observeSnapshot(
                text: "我们配合 Ghostty 来实现 Terminal CLI 的输入。",
                elapsedSinceLastChange: AutomaticDictionaryLearningMonitor.idleSettleSeconds + 0.1,
                state: &state
            ),
            .settleForAnalysis(finalText: "我们配合 Ghostty 来实现 Terminal CLI 的输入。")
        )
    }

    func testObservationRegistersNewInputChangeAndResetsMissingCounter() {
        var state = AutomaticDictionaryLearningObservationState(
            baselineText: "baseline"
        )
        state.consecutiveMissingSnapshots = 2

        let decision = AutomaticDictionaryLearningMonitor.observeSnapshot(
            text: "baseline updated",
            elapsedSinceLastChange: nil,
            state: &state
        )

        XCTAssertEqual(decision, .continueObserving)
        XCTAssertEqual(state.latestText, "baseline updated")
        XCTAssertTrue(state.didObserveChange)
        XCTAssertEqual(state.consecutiveMissingSnapshots, 0)
        XCTAssertEqual(state.lastChangeElapsedSeconds, 0)
    }

    func testRejectsPureAppendAfterInsertion() {
        let outcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: "hello",
            baselineText: "hello",
            finalText: "hello world"
        )

        assertSkipped(outcome, contains: "does not intersect inserted text")
    }

    func testRejectsUnrelatedEditsOutsideInsertedText() {
        let outcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: "Anthropic",
            baselineText: "Anthropic works. tomorrow 3pm",
            finalText: "Anthropic works. tomorrow 4pm"
        )

        assertSkipped(outcome, contains: "does not intersect inserted text")
    }

    func testRejectsLargeRewrite() {
        let outcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: "short note",
            baselineText: "short note",
            finalText: "Completely different long paragraph with multiple rewritten clauses and unrelated content."
        )

        assertSkipped(outcome, contains: "edit ratio")
    }

    func testBuildPromptResolvesTemplateVariables() {
        let request = AutomaticDictionaryLearningRequest(
            insertedText: "anthropic ai",
            baselineContext: "Please ship anthropic ai today.",
            finalContext: "Please ship Anthropic today.",
            baselineChangedFragment: "anthropic ai",
            finalChangedFragment: "Anthropic",
            editRatio: 0.2
        )

        let prompt = AutomaticDictionaryLearningMonitor.buildPrompt(
            template: """
            \(AppPreferenceKey.automaticDictionaryLearningMainLanguageTemplateVariable)
            \(AppPreferenceKey.automaticDictionaryLearningOtherLanguagesTemplateVariable)
            \(AppPreferenceKey.automaticDictionaryLearningInsertedTextTemplateVariable)
            \(AppPreferenceKey.automaticDictionaryLearningFinalFragmentTemplateVariable)
            \(AppPreferenceKey.automaticDictionaryLearningExistingTermsTemplateVariable)
            """,
            for: request,
            existingTerms: ["OpenAI", "Claude"],
            userMainLanguage: "Chinese",
            userOtherLanguages: "English"
        )

        XCTAssertTrue(prompt.contains("Chinese"))
        XCTAssertTrue(prompt.contains("English"))
        XCTAssertTrue(prompt.contains("anthropic ai"))
        XCTAssertTrue(prompt.contains("Anthropic"))
        XCTAssertTrue(prompt.contains("- OpenAI"))
        XCTAssertTrue(prompt.contains("- Claude"))
        XCTAssertFalse(prompt.contains(AppPreferenceKey.automaticDictionaryLearningExistingTermsTemplateVariable))
    }

    func testBuildPromptUsesEmptyPlaceholderForExistingTerms() {
        let request = AutomaticDictionaryLearningRequest(
            insertedText: "Voxt",
            baselineContext: "Voxt",
            finalContext: "Voxt",
            baselineChangedFragment: "vox",
            finalChangedFragment: "Voxt",
            editRatio: 0.1
        )

        let prompt = AutomaticDictionaryLearningMonitor.buildPrompt(
            template: AppPreferenceKey.automaticDictionaryLearningExistingTermsTemplateVariable,
            for: request,
            existingTerms: [],
            userMainLanguage: "Chinese",
            userOtherLanguages: "English"
        )

        XCTAssertTrue(prompt.contains("(empty)"))
    }

    func testBuildPromptIncludesExtractedCandidateTerms() {
        let request = AutomaticDictionaryLearningRequest(
            insertedText: "这次换成J W T加瑞的斯",
            baselineContext: "这次换成J W T加瑞的斯",
            finalContext: "这次换成JWT + Redis",
            baselineChangedFragment: "J W T加瑞的斯",
            finalChangedFragment: "JWT + Redis",
            editRatio: 0.2
        )

        let prompt = AutomaticDictionaryLearningMonitor.buildPrompt(
            template: "Review the correction.",
            for: request,
            existingTerms: [],
            userMainLanguage: "Chinese",
            userOtherLanguages: "English"
        )

        XCTAssertTrue(prompt.contains("<candidate_terms>\nJWT, Redis\n</candidate_terms>"))
    }

    func testBuildPromptCapsExistingTermListToTwentyItems() {
        let request = AutomaticDictionaryLearningRequest(
            insertedText: "Voxt",
            baselineContext: "Voxt",
            finalContext: "Voxt",
            baselineChangedFragment: "vox",
            finalChangedFragment: "Voxt",
            editRatio: 0.1
        )

        let existingTerms = (1...100).map { "Term\($0)" }
        let prompt = AutomaticDictionaryLearningMonitor.buildPrompt(
            template: AppPreferenceKey.automaticDictionaryLearningExistingTermsTemplateVariable,
            for: request,
            existingTerms: existingTerms,
            userMainLanguage: "Chinese",
            userOtherLanguages: "English"
        )

        XCTAssertTrue(prompt.contains("- Term1"))
        XCTAssertTrue(prompt.contains("- Term20"))
        XCTAssertFalse(prompt.contains("- Term21"))
    }

    private func assertSkipped(
        _ outcome: AutomaticDictionaryLearningMonitor.RequestOutcome,
        contains expectedReasonFragment: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard case .skipped(let reason) = outcome else {
            return XCTFail("Expected skipped outcome, got \(outcome)", file: file, line: line)
        }
        XCTAssertTrue(
            reason.contains(expectedReasonFragment),
            "Expected reason to contain '\(expectedReasonFragment)', got '\(reason)'",
            file: file,
            line: line
        )
    }

    private func assertLearningDiff(
        insertedText: String,
        finalText: String,
        expectedBaselineFragment: String,
        expectedFinalFragment: String,
        message: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let outcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: insertedText,
            baselineText: insertedText,
            finalText: finalText
        )

        guard case .ready(let request) = outcome else {
            return XCTFail("Expected ready outcome for \(message), got \(outcome)", file: file, line: line)
        }

        XCTAssertTrue(
            fragmentsOverlap(
                request.baselineChangedFragment,
                expectedBaselineFragment
            ),
            "Unexpected baseline fragment for \(message): \(request.baselineChangedFragment)",
            file: file,
            line: line
        )
        XCTAssertTrue(
            fragmentsOverlap(
                request.finalChangedFragment,
                expectedFinalFragment
            ),
            "Unexpected final fragment for \(message): \(request.finalChangedFragment)",
            file: file,
            line: line
        )
        XCTAssertLessThanOrEqual(
            request.editRatio,
            AutomaticDictionaryLearningMonitor.maximumEditRatio,
            "Unexpected edit ratio for \(message)",
            file: file,
            line: line
        )
    }

    private func fragmentsOverlap(_ actual: String, _ expected: String) -> Bool {
        actual.contains(expected) || expected.contains(actual)
    }
}
