package server

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestOllamaMessagesToOpenAIToolResponseBinding covers the /api/chat path:
// validateToolRequest, then ollamaMessagesToOpenAI.
func TestOllamaMessagesToOpenAIToolResponseBinding(t *testing.T) {
	call := func(id, name string) OllamaToolCall {
		return OllamaToolCall{ID: id, Function: OllamaToolCallFunc{Name: name, Arguments: map[string]interface{}{}}}
	}
	asst := func(calls ...OllamaToolCall) OllamaMessage {
		return OllamaMessage{Role: "assistant", ToolCalls: calls}
	}
	tool := func(id, name, content string) OllamaMessage {
		return OllamaMessage{Role: "tool", ToolCallID: id, ToolName: name, Content: content}
	}
	user := OllamaMessage{Role: "user", Content: "hi"}

	toolCallIDs := func(msgs []map[string]interface{}) []string {
		ids := []string{}
		for _, m := range msgs {
			if m["role"] == "tool" {
				id, _ := m["tool_call_id"].(string)
				ids = append(ids, id)
			}
		}
		return ids
	}

	tests := []struct {
		name    string
		in      []OllamaMessage
		wantIDs []string
		wantLen int
	}{
		{
			name:    "identifierless response binds to the actual call id",
			in:      []OllamaMessage{user, asst(call("call_abc", "bash")), tool("", "", "/home/user")},
			wantIDs: []string{"call_abc"},
			wantLen: 3,
		},
		{
			name:    "multiple identifierless responses bind by position",
			in:      []OllamaMessage{user, asst(call("call_A", "a"), call("call_B", "b")), tool("", "", "ra"), tool("", "", "rb")},
			wantIDs: []string{"call_A", "call_B"},
			wantLen: 4,
		},
		{
			name:    "explicit id consumes its slot so the next identifierless response gets the remaining call",
			in:      []OllamaMessage{user, asst(call("call_A", "a"), call("call_B", "b")), tool("call_A", "", "ra"), tool("", "", "rb")},
			wantIDs: []string{"call_A", "call_B"},
			wantLen: 4,
		},
		{
			name: "tool_name binds by function name even out of order",
			in: []OllamaMessage{user, asst(call("call_A", "get_weather"), call("call_B", "get_time")),
				tool("", "get_time", "12:00"), tool("", "get_weather", "sunny")},
			wantIDs: []string{"call_B", "call_A"},
			wantLen: 4,
		},
		{
			name: "unbindable response after all calls consumed is dropped",
			in: []OllamaMessage{user, asst(call("call_A", "a")),
				tool("", "", "ra"), tool("", "", "response to hallucinated call")},
			wantIDs: []string{"call_A"},
			wantLen: 3,
		},
		{
			name: "stale round does not capture a later orphan response",
			in: []OllamaMessage{user, asst(call("call_A", "a")), tool("call_A", "", "ra"),
				{Role: "assistant", Content: "done"}, tool("", "", "orphan")},
			wantIDs: []string{"call_A"},
			wantLen: 4,
		},
		{
			name:    "named response with no pending calls keeps a synthetic id",
			in:      []OllamaMessage{user, tool("", "foo", "x")},
			wantIDs: []string{"call_foo_1"},
			wantLen: 2,
		},
		{
			name:    "explicit ids are always preserved verbatim",
			in:      []OllamaMessage{user, asst(call("call_A", "a")), tool("call_XYZ", "", "ra")},
			wantIDs: []string{"call_XYZ"},
			wantLen: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &OllamaChatRequest{Messages: append([]OllamaMessage(nil), tt.in...)}
			assert.NoError(t, validateToolRequest(req))
			out := ollamaMessagesToOpenAI(req.Messages)
			assert.Len(t, out, tt.wantLen)
			assert.Equal(t, tt.wantIDs, toolCallIDs(out))
		})
	}
}
