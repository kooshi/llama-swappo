package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/mostlygeek/llama-swap/proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestAaaOllamaToolCallingFlow runs all tool calling tests using direct handler testing
func TestAaaOllamaToolCallingFlow(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Create a mock backend that handles tool calls
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		// Check if request has tools (initial request with tools)
		if tools, ok := req["tools"].([]interface{}); ok && len(tools) > 0 {
			// Return tool call response
			response := map[string]interface{}{
				"id":      "chatcmpl-123",
				"object":  "chat.completion",
				"created": 1677652288,
				"model":   "test-model",
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"message": map[string]interface{}{
							"role":    "assistant",
							"content": nil,
							"tool_calls": []map[string]interface{}{
								{
									"id":   "call_123",
									"type": "function",
									"function": map[string]interface{}{
										"name":      "get_weather",
										"arguments": `{"location":"test location"}`,
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
			return
		}

		// Check messages for tool responses
		if messages, ok := req["messages"].([]interface{}); ok {
			for _, msg := range messages {
				if msgMap, ok := msg.(map[string]interface{}); ok {
					if role, ok := msgMap["role"].(string); ok && role == "tool" {
						// Found a tool response, return final response
						response := map[string]interface{}{
							"id":      "chatcmpl-123",
							"object":  "chat.completion",
							"created": 1677652288,
							"model":   "test-model",
							"choices": []map[string]interface{}{
								{
									"index": 0,
									"message": map[string]interface{}{
										"role":    "assistant",
										"content": "The weather in Boston is 72 degrees and sunny.",
									},
									"finish_reason": "stop",
								},
							},
						}
						w.Header().Set("Content-Type", "application/json")
						json.NewEncoder(w).Encode(response)
						return
					}
				}
			}
		}

		// Default response for normal chat
		response := map[string]interface{}{
			"id":      "chatcmpl-123",
			"object":  "chat.completion",
			"created": 1677652288,
			"model":   "test-model",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Hello! How can I help you today?",
					},
					"finish_reason": "stop",
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer backend.Close()

	// Create proxy config with mock backend
	cfg := config.Config{
		Models: map[string]config.ModelConfig{
			"test-model": {
				Cmd:           "sleep 3600", // Long-running command
				Proxy:         backend.URL,
				CheckEndpoint: "none", // Skip health check
			},
		},
	}
	cfg = config.AddDefaultGroupToConfig(cfg)

	// Create ProxyManager manually without registering routes to avoid conflicts
	pm := &ProxyManager{}
	pm.config = cfg
	pm.proxyLogger = testLogger
	pm.processGroups = make(map[string]*ProcessGroup)

	// Initialize process groups (needed for model lookup)
	for groupID := range cfg.Groups {
		processGroup := NewProcessGroup(groupID, cfg, testLogger, testLogger)
		pm.processGroups[groupID] = processGroup
	}

	t.Run("ToolCallRequest", func(t *testing.T) {
		req := OllamaChatRequest{
			Model: "test-model",
			Messages: []OllamaMessage{
				{
					Role:    "user",
					Content: "What's the weather in Boston?",
				},
			},
			Tools: []OllamaTool{
				{
					Type: "function",
					Function: OllamaToolFunction{
						Name:        "get_weather",
						Description: "Get weather for a location",
						Parameters: map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"location": map[string]interface{}{
									"type": "string",
								},
							},
							"required": []string{"location"},
						},
					},
				},
			},
			Stream: boolPtr(false),
		}

		reqBodyBytes, _ := json.Marshal(req)
		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBuffer(reqBodyBytes))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		assert.Equal(t, http.StatusOK, w.Code)

		var resp OllamaChatResponse
		err := json.Unmarshal(w.Body.Bytes(), &resp)
		require.NoError(t, err)

		// Verify response structure
		assert.Equal(t, "test-model", resp.Model)
		assert.Equal(t, "assistant", resp.Message.Role)
		assert.Equal(t, "tool_calls", resp.DoneReason)
		assert.True(t, resp.Done)

		// Verify tool calls are in the message
		require.Len(t, resp.Message.ToolCalls, 1, "Should have one tool call")
		toolCall := resp.Message.ToolCalls[0]
		assert.Equal(t, "call_123", toolCall.ID)
		assert.Equal(t, "function", toolCall.Type)
		assert.Equal(t, "get_weather", toolCall.Function.Name)

		// Arguments is already a map, just check it directly
		args := toolCall.Function.Arguments
		assert.Equal(t, "test location", args["location"])
	})

	t.Run("NormalRequest", func(t *testing.T) {
		req := OllamaChatRequest{
			Model: "test-model",
			Messages: []OllamaMessage{
				{
					Role:    "user",
					Content: "Hello",
				},
			},
			Stream: boolPtr(false),
		}

		reqBodyBytes, _ := json.Marshal(req)
		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBuffer(reqBodyBytes))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		assert.Equal(t, http.StatusOK, w.Code)

		var resp OllamaChatResponse
		err := json.Unmarshal(w.Body.Bytes(), &resp)
		require.NoError(t, err)

		assert.Equal(t, "test-model", resp.Model)
		assert.Equal(t, "assistant", resp.Message.Role)
		assert.Equal(t, "Hello! How can I help you today?", resp.Message.Content)
		assert.Empty(t, resp.Message.ToolCalls)
	})

	t.Run("ToolResponseMessages", func(t *testing.T) {
		// Test with tool response message
		req := OllamaChatRequest{
			Model: "test-model",
			Messages: []OllamaMessage{
				{
					Role:    "user",
					Content: "What's the weather in Boston?",
				},
				{
					Role:    "assistant",
					Content: "I'll check the weather for you.",
					ToolCalls: []OllamaToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: OllamaToolCallFunc{
								Index: 0,
								Name:  "get_weather",
								Arguments: map[string]interface{}{
									"location": "Boston",
								},
							},
						},
					},
				},
				{
					Role:       "tool",
					ToolCallID: "call_123",
					Content:    "72 degrees and sunny",
				},
			},
			Stream: boolPtr(false),
		}

		reqBodyBytes, _ := json.Marshal(req)
		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBuffer(reqBodyBytes))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		assert.Equal(t, http.StatusOK, w.Code)

		var resp OllamaChatResponse
		err := json.Unmarshal(w.Body.Bytes(), &resp)
		require.NoError(t, err)

		assert.Equal(t, "test-model", resp.Model)
		assert.Contains(t, resp.Message.Content, "72 degrees and sunny")
		assert.Empty(t, resp.Message.ToolCalls) // No new tool calls in the final response
	})
}

// Helper function
func boolPtr(b bool) *bool {
	return &b
}

// TestZedStyleToolCalls tests that Zed's Ollama client format is accepted
// Zed sends tool_calls without id/type fields at the top level
func TestZedStyleToolCalls(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Create a mock backend
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"id":      "chatcmpl-123",
			"object":  "chat.completion",
			"created": 1677652288,
			"model":   "test-model",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "The weather is nice.",
					},
					"finish_reason": "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer backend.Close()

	cfg := config.Config{
		Models: map[string]config.ModelConfig{
			"test-model": {
				Cmd:           "sleep 3600",
				Proxy:         backend.URL,
				CheckEndpoint: "none",
			},
		},
	}
	cfg = config.AddDefaultGroupToConfig(cfg)

	pm := &ProxyManager{}
	pm.config = cfg
	pm.proxyLogger = testLogger
	pm.processGroups = make(map[string]*ProcessGroup)

	for groupID := range cfg.Groups {
		processGroup := NewProcessGroup(groupID, cfg, testLogger, testLogger)
		pm.processGroups[groupID] = processGroup
	}

	t.Run("ZedStyleNoIdNoType", func(t *testing.T) {
		// Zed sends tool_calls with only function field, no id or type
		// This is the raw JSON that Zed's OllamaToolCall serializes to
		rawJSON := `{
			"model": "test-model",
			"stream": false,
			"messages": [
				{"role": "user", "content": "What is the weather?"},
				{
					"role": "assistant",
					"content": "",
					"tool_calls": [
						{
							"function": {
								"name": "get_weather",
								"arguments": {"location": "Paris"}
							}
						}
					]
				},
				{
					"role": "tool",
					"tool_call_id": "call_0_0",
					"content": "Sunny, 22C"
				}
			],
			"tools": [
				{
					"type": "function",
					"function": {
						"name": "get_weather",
						"description": "Get weather",
						"parameters": {
							"type": "object",
							"properties": {
								"location": {"type": "string"}
							}
						}
					}
				}
			]
		}`

		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBufferString(rawJSON))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		// Should NOT return 400 Bad Request
		assert.Equal(t, http.StatusOK, w.Code, "Response body: %s", w.Body.String())
	})

	t.Run("OllamaNativeFormat", func(t *testing.T) {
		// Ollama native format from the docs - no id, no type, just function
		rawJSON := `{
			"model": "test-model",
			"stream": false,
			"messages": [
				{"role": "user", "content": "what is the weather in Toronto?"},
				{
					"role": "assistant",
					"tool_calls": [
						{
							"function": {
								"name": "get_temperature",
								"arguments": {"city": "Toronto"}
							}
						}
					]
				},
				{
					"role": "tool",
					"content": "11 degrees celsius",
					"tool_call_id": "call_0_0"
				}
			],
			"tools": [
				{
					"type": "function",
					"function": {
						"name": "get_temperature",
						"description": "Get temperature",
						"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
					}
				}
			]
		}`

		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBufferString(rawJSON))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		assert.Equal(t, http.StatusOK, w.Code, "Response body: %s", w.Body.String())
	})

	t.Run("MultipleToolCallsNoIds", func(t *testing.T) {
		// Multiple parallel tool calls without IDs
		rawJSON := `{
			"model": "test-model",
			"stream": false,
			"messages": [
				{"role": "user", "content": "weather in Paris and London?"},
				{
					"role": "assistant",
					"content": "",
					"tool_calls": [
						{"function": {"name": "get_weather", "arguments": {"location": "Paris"}}},
						{"function": {"name": "get_weather", "arguments": {"location": "London"}}}
					]
				},
				{"role": "tool", "tool_call_id": "call_1_0", "content": "Paris: Sunny"},
				{"role": "tool", "tool_call_id": "call_1_1", "content": "London: Rainy"}
			],
			"tools": [
				{
					"type": "function",
					"function": {
						"name": "get_weather",
						"description": "Get weather",
						"parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
					}
				}
			]
		}`

		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBufferString(rawJSON))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		assert.Equal(t, http.StatusOK, w.Code, "Response body: %s", w.Body.String())
	})

	t.Run("OllamaNativeToolName", func(t *testing.T) {
		// Ollama native format uses tool_name instead of tool_call_id for tool responses
		// This is what Zed sends when using Ollama's native format
		rawJSON := `{
			"model": "test-model",
			"stream": false,
			"messages": [
				{"role": "user", "content": "find something"},
				{
					"role": "assistant",
					"content": "",
					"tool_calls": [
						{"id": "call_1", "function": {"name": "find_path", "arguments": {"glob": "*.txt"}}}
					]
				},
				{"role": "tool", "tool_name": "find_path", "content": "file.txt"}
			],
			"tools": [
				{
					"type": "function",
					"function": {
						"name": "find_path",
						"description": "Find files",
						"parameters": {"type": "object", "properties": {"glob": {"type": "string"}}}
					}
				}
			]
		}`

		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBufferString(rawJSON))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		// Should succeed - tool_name is a valid alternative to tool_call_id
		assert.Equal(t, http.StatusOK, w.Code, "Response body: %s", w.Body.String())
	})

	t.Run("HallucinatedEmptyToolCallsFiltered", func(t *testing.T) {
		// Some local models hallucinate empty tool calls alongside valid ones
		// These should be filtered out, not cause errors
		rawJSON := `{
			"model": "test-model",
			"stream": false,
			"messages": [
				{"role": "user", "content": "find files"},
				{
					"role": "assistant",
					"content": "",
					"tool_calls": [
						{"id": "call_valid", "function": {"name": "find_path", "arguments": {"glob": "*.go"}}},
						{"id": "call_empty1", "function": {"name": "", "arguments": null}},
						{"id": "call_empty2", "function": {"name": "", "arguments": {}}}
					]
				},
				{"role": "tool", "tool_call_id": "call_valid", "content": "main.go\ntest.go"}
			],
			"tools": [
				{
					"type": "function",
					"function": {
						"name": "find_path",
						"description": "Find files",
						"parameters": {"type": "object", "properties": {"glob": {"type": "string"}}}
					}
				}
			]
		}`

		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBufferString(rawJSON))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		// Should succeed - empty tool calls filtered, valid one processed
		assert.Equal(t, http.StatusOK, w.Code, "Response body: %s", w.Body.String())
	})

}

// TestStreamingToolCallAccumulation tests that tool calls streamed incrementally
// by OpenAI are properly accumulated and emitted as complete tool calls in Ollama format
func TestStreamingToolCallAccumulation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Create a mock backend that streams tool calls incrementally like OpenAI does
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]interface{}
		json.NewDecoder(r.Body).Decode(&req)

		// Check if streaming is requested
		stream, _ := req["stream"].(bool)
		if !stream {
			t.Fatal("Expected streaming request")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher := w.(http.Flusher)

		// Simulate OpenAI's incremental streaming of tool calls:
		// 1. First chunk: tool call ID and type
		// 2. Second chunk: function name
		// 3. Third+ chunks: arguments as string fragments
		// 4. Final chunk: finish_reason

		chunks := []string{
			// Chunk 1: ID and type for first tool call
			`{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"","arguments":""}}]},"finish_reason":null}]}`,
			// Chunk 2: Function name
			`{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
			// Chunk 3: Arguments fragment 1
			`{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":"}}]},"finish_reason":null}]}`,
			// Chunk 4: Arguments fragment 2
			`{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Boston\"}"}}]},"finish_reason":null}]}`,
			// Chunk 5: Second tool call starts
			`{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_def456","type":"function","function":{"name":"get_time","arguments":""}}]},"finish_reason":null}]}`,
			// Chunk 6: Second tool call arguments
			`{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"timezone\":\"EST\"}"}}]},"finish_reason":null}]}`,
			// Chunk 7: Final chunk with finish_reason
			`{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		}

		for _, chunk := range chunks {
			w.Write([]byte("data: " + chunk + "\n\n"))
			flusher.Flush()
		}
		w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer backend.Close()

	cfg := config.Config{
		Models: map[string]config.ModelConfig{
			"test-model": {
				Cmd:           "sleep 3600",
				Proxy:         backend.URL,
				CheckEndpoint: "none",
			},
		},
	}
	cfg = config.AddDefaultGroupToConfig(cfg)

	pm := &ProxyManager{}
	pm.config = cfg
	pm.proxyLogger = testLogger
	pm.processGroups = make(map[string]*ProcessGroup)

	for groupID := range cfg.Groups {
		processGroup := NewProcessGroup(groupID, cfg, testLogger, testLogger)
		pm.processGroups[groupID] = processGroup
	}

	t.Run("AccumulatesStreamingToolCalls", func(t *testing.T) {
		req := OllamaChatRequest{
			Model: "test-model",
			Messages: []OllamaMessage{
				{Role: "user", Content: "What's the weather and time in Boston?"},
			},
			Tools: []OllamaTool{
				{
					Type: "function",
					Function: OllamaToolFunction{
						Name:        "get_weather",
						Description: "Get weather",
						Parameters:  map[string]interface{}{"type": "object", "properties": map[string]interface{}{"location": map[string]interface{}{"type": "string"}}},
					},
				},
				{
					Type: "function",
					Function: OllamaToolFunction{
						Name:        "get_time",
						Description: "Get time",
						Parameters:  map[string]interface{}{"type": "object", "properties": map[string]interface{}{"timezone": map[string]interface{}{"type": "string"}}},
					},
				},
			},
			Stream: boolPtr(true),
		}

		reqBodyBytes, _ := json.Marshal(req)
		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBuffer(reqBodyBytes))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		assert.Equal(t, http.StatusOK, w.Code)

		// Parse the streaming response - collect all JSON lines
		body := w.Body.String()
		lines := bytes.Split([]byte(body), []byte("\n"))

		var toolCallResp *OllamaChatResponse
		var doneResp *OllamaChatResponse
		toolCallChunks := 0

		for _, line := range lines {
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}

			var resp OllamaChatResponse
			if err := json.Unmarshal(line, &resp); err != nil {
				continue
			}

			// Count chunks that have tool calls (should only appear once)
			if len(resp.Message.ToolCalls) > 0 {
				toolCallChunks++
				respCopy := resp
				toolCallResp = &respCopy
			}
			// Track the final done:true chunk
			if resp.Done {
				respCopy := resp
				doneResp = &respCopy
			}
		}

		// Key assertion: tool calls should only appear ONCE
		// Not in every intermediate chunk
		assert.Equal(t, 1, toolCallChunks, "Tool calls should only appear once, not streamed incrementally")

		// Verify the accumulated tool calls are complete and correct
		require.NotNil(t, toolCallResp, "Should have a response with tool calls")
		require.Len(t, toolCallResp.Message.ToolCalls, 2, "Should have 2 accumulated tool calls")

		// First tool call
		tc1 := toolCallResp.Message.ToolCalls[0]
		assert.Equal(t, "call_abc123", tc1.ID)
		assert.Equal(t, "get_weather", tc1.Function.Name)
		assert.Equal(t, "Boston", tc1.Function.Arguments["location"])

		// Second tool call
		tc2 := toolCallResp.Message.ToolCalls[1]
		assert.Equal(t, "call_def456", tc2.ID)
		assert.Equal(t, "get_time", tc2.Function.Name)
		assert.Equal(t, "EST", tc2.Function.Arguments["timezone"])

		// Ollama-compatible: tool_calls chunk has done:false
		assert.False(t, toolCallResp.Done, "Chunk with tool_calls should have done:false")
		assert.Equal(t, "tool_calls", toolCallResp.DoneReason)

		// Verify there's a separate done:true chunk
		require.NotNil(t, doneResp, "Should have a final done:true chunk")
		assert.True(t, doneResp.Done)
		assert.Empty(t, doneResp.Message.ToolCalls, "Final done:true chunk should not have tool_calls")
	})
}

// TestStreamingChatResponses tests streaming chat responses without tool calls
func TestStreamingChatResponses(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("StreamingTextResponse", func(t *testing.T) {
		// Mock backend that streams a text response
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			flusher := w.(http.Flusher)

			chunks := []string{
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
				flusher.Flush()
			}
			w.Write([]byte("data: [DONE]\n\n"))
			flusher.Flush()
		}))
		defer backend.Close()

		cfg := config.Config{
			Models: map[string]config.ModelConfig{
				"test-model": {Cmd: "sleep 3600", Proxy: backend.URL, CheckEndpoint: "none"},
			},
		}
		cfg = config.AddDefaultGroupToConfig(cfg)

		pm := &ProxyManager{config: cfg, proxyLogger: testLogger, processGroups: make(map[string]*ProcessGroup)}
		for groupID := range cfg.Groups {
			pm.processGroups[groupID] = NewProcessGroup(groupID, cfg, testLogger, testLogger)
		}

		reqBody := `{"model": "test-model", "stream": true, "messages": [{"role": "user", "content": "Hi"}]}`
		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBufferString(reqBody))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		assert.Equal(t, http.StatusOK, w.Code)

		// Parse streaming response and collect content
		body := w.Body.String()
		lines := bytes.Split([]byte(body), []byte("\n"))

		var contentParts []string
		var finalResp *OllamaChatResponse

		for _, line := range lines {
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}

			var resp OllamaChatResponse
			if err := json.Unmarshal(line, &resp); err != nil {
				continue
			}

			if resp.Message.Content != "" {
				contentParts = append(contentParts, resp.Message.Content)
			}
			if resp.Done {
				finalResp = &resp
			}
		}

		// Verify content was streamed in parts
		assert.Equal(t, []string{"Hello", " there", "!"}, contentParts)
		require.NotNil(t, finalResp)
		assert.True(t, finalResp.Done)
		assert.Equal(t, "stop", finalResp.DoneReason)
	})

	t.Run("StreamingSingleToolCall", func(t *testing.T) {
		// Mock backend that streams a single tool call
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			flusher := w.(http.Flusher)

			chunks := []string{
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"NYC\"}"}}]},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
				flusher.Flush()
			}
			w.Write([]byte("data: [DONE]\n\n"))
			flusher.Flush()
		}))
		defer backend.Close()

		cfg := config.Config{
			Models: map[string]config.ModelConfig{
				"test-model": {Cmd: "sleep 3600", Proxy: backend.URL, CheckEndpoint: "none"},
			},
		}
		cfg = config.AddDefaultGroupToConfig(cfg)

		pm := &ProxyManager{config: cfg, proxyLogger: testLogger, processGroups: make(map[string]*ProcessGroup)}
		for groupID := range cfg.Groups {
			pm.processGroups[groupID] = NewProcessGroup(groupID, cfg, testLogger, testLogger)
		}

		reqBody := `{
			"model": "test-model",
			"stream": true,
			"messages": [{"role": "user", "content": "weather in NYC?"}],
			"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
		}`
		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBufferString(reqBody))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		assert.Equal(t, http.StatusOK, w.Code)

		// Find the response with tool calls (Ollama-compat: done:false)
		body := w.Body.String()
		lines := bytes.Split([]byte(body), []byte("\n"))

		var toolCallResp *OllamaChatResponse
		for _, line := range lines {
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}
			var resp OllamaChatResponse
			if err := json.Unmarshal(line, &resp); err != nil {
				continue
			}
			if len(resp.Message.ToolCalls) > 0 {
				respCopy := resp
				toolCallResp = &respCopy
			}
		}

		require.NotNil(t, toolCallResp, "Should have response with tool calls")
		require.Len(t, toolCallResp.Message.ToolCalls, 1)
		assert.Equal(t, "call_1", toolCallResp.Message.ToolCalls[0].ID)
		assert.Equal(t, "get_weather", toolCallResp.Message.ToolCalls[0].Function.Name)
		assert.Equal(t, "NYC", toolCallResp.Message.ToolCalls[0].Function.Arguments["city"])
		// Ollama-compat: tool_calls chunk has done:false
		assert.False(t, toolCallResp.Done, "Tool calls chunk should have done:false")
	})

	t.Run("StreamingWithHallucinatedToolCallsFiltered", func(t *testing.T) {
		// Mock backend that streams tool calls including hallucinated empty ones
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			flusher := w.(http.Flusher)

			chunks := []string{
				// Valid tool call
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_valid","type":"function","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":\"test\"}"}}]},"finish_reason":null}]}`,
				// Hallucinated empty tool call (no name)
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_bad","type":"function","function":{"name":"","arguments":""}}]},"finish_reason":null}]}`,
				// Final
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
				flusher.Flush()
			}
			w.Write([]byte("data: [DONE]\n\n"))
			flusher.Flush()
		}))
		defer backend.Close()

		cfg := config.Config{
			Models: map[string]config.ModelConfig{
				"test-model": {Cmd: "sleep 3600", Proxy: backend.URL, CheckEndpoint: "none"},
			},
		}
		cfg = config.AddDefaultGroupToConfig(cfg)

		pm := &ProxyManager{config: cfg, proxyLogger: testLogger, processGroups: make(map[string]*ProcessGroup)}
		for groupID := range cfg.Groups {
			pm.processGroups[groupID] = NewProcessGroup(groupID, cfg, testLogger, testLogger)
		}

		reqBody := `{
			"model": "test-model",
			"stream": true,
			"messages": [{"role": "user", "content": "search"}],
			"tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}]
		}`
		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBufferString(reqBody))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		assert.Equal(t, http.StatusOK, w.Code)

		// Find response with tool calls (Ollama-compat: done:false)
		body := w.Body.String()
		lines := bytes.Split([]byte(body), []byte("\n"))

		var toolCallResp *OllamaChatResponse
		for _, line := range lines {
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}
			var resp OllamaChatResponse
			if err := json.Unmarshal(line, &resp); err != nil {
				continue
			}
			if len(resp.Message.ToolCalls) > 0 {
				respCopy := resp
				toolCallResp = &respCopy
			}
		}

		require.NotNil(t, toolCallResp)
		// Should only have the valid tool call, empty one filtered
		require.Len(t, toolCallResp.Message.ToolCalls, 1, "Hallucinated empty tool call should be filtered")
		assert.Equal(t, "search", toolCallResp.Message.ToolCalls[0].Function.Name)
		// Ollama-compat: tool_calls chunk has done:false
		assert.False(t, toolCallResp.Done, "Tool calls chunk should have done:false")
	})
}

// TestStreamingToolCallsOllamaCompatible tests that streaming tool calls follow
// Ollama's native format: tool_calls in a chunk with done:false, then a final
// chunk with done:true (no tool_calls). This matches how Ollama proper behaves
// and ensures compatibility with clients like rig that process done flag first.
func TestStreamingToolCallsOllamaCompatible(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Mock backend that streams tool calls like OpenAI
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher := w.(http.Flusher)

		chunks := []string{
			// Tool call ID and name
			`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
			// Arguments
			`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":\"Boston\"}"}}]},"finish_reason":null}]}`,
			// Final chunk with finish_reason
			`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		}

		for _, chunk := range chunks {
			w.Write([]byte("data: " + chunk + "\n\n"))
			flusher.Flush()
		}
		w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer backend.Close()

	cfg := config.Config{
		Models: map[string]config.ModelConfig{
			"test-model": {Cmd: "sleep 3600", Proxy: backend.URL, CheckEndpoint: "none"},
		},
	}
	cfg = config.AddDefaultGroupToConfig(cfg)

	pm := &ProxyManager{config: cfg, proxyLogger: testLogger, processGroups: make(map[string]*ProcessGroup)}
	for groupID := range cfg.Groups {
		pm.processGroups[groupID] = NewProcessGroup(groupID, cfg, testLogger, testLogger)
	}

	t.Run("ToolCallsInSeparateChunkBeforeDone", func(t *testing.T) {
		reqBody := `{
			"model": "test-model",
			"stream": true,
			"messages": [{"role": "user", "content": "weather?"}],
			"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
		}`
		httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewBufferString(reqBody))
		httpReq.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httpReq

		pm.ollamaChatHandler()(c)

		assert.Equal(t, http.StatusOK, w.Code)

		// Parse all chunks
		body := w.Body.String()
		lines := bytes.Split([]byte(body), []byte("\n"))

		var allResponses []OllamaChatResponse
		for _, line := range lines {
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}
			var resp OllamaChatResponse
			if err := json.Unmarshal(line, &resp); err != nil {
				continue
			}
			allResponses = append(allResponses, resp)
		}

		require.GreaterOrEqual(t, len(allResponses), 2, "Should have at least 2 chunks")

		// Find the chunk with tool_calls and the final done chunk
		var toolCallChunk *OllamaChatResponse
		var finalChunk *OllamaChatResponse

		for i := range allResponses {
			resp := &allResponses[i]
			if len(resp.Message.ToolCalls) > 0 {
				toolCallChunk = resp
			}
			if resp.Done {
				finalChunk = resp
			}
		}

		// KEY ASSERTION: Tool calls should be in a chunk with done:false
		require.NotNil(t, toolCallChunk, "Should have a chunk with tool_calls")
		assert.False(t, toolCallChunk.Done, "Chunk with tool_calls should have done:false (Ollama-compatible)")
		assert.Equal(t, "tool_calls", toolCallChunk.DoneReason, "Should have done_reason even with done:false")

		// Verify tool call content
		require.Len(t, toolCallChunk.Message.ToolCalls, 1)
		assert.Equal(t, "call_123", toolCallChunk.Message.ToolCalls[0].ID)
		assert.Equal(t, "get_weather", toolCallChunk.Message.ToolCalls[0].Function.Name)
		assert.Equal(t, "Boston", toolCallChunk.Message.ToolCalls[0].Function.Arguments["location"])

		// KEY ASSERTION: Final chunk should have done:true but NO tool_calls
		require.NotNil(t, finalChunk, "Should have a final chunk with done:true")
		assert.True(t, finalChunk.Done, "Final chunk should have done:true")
		assert.Empty(t, finalChunk.Message.ToolCalls, "Final done:true chunk should NOT contain tool_calls")
	})
}
