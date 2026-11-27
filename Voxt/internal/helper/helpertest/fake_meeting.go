package helpertest

import "github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"

func (f *fake) meetingWindow(message proto.Message) {
	if message.Method != proto.MethodMeetingWindow || f.has("no-output") {
		f.fail(message.ID, proto.CodeMethodNotFound, "Unknown method %s", message.Method)
		return
	}
	f.result(message.ID, proto.MeetingWindowResult{Nodes: []proto.MeetingNode{}})
}
