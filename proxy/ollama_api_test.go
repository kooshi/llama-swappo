package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

// TestNormalizeKeepAlive tests the normalizeKeepAlive helper function
func TestNormalizeKeepAlive(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected string
	}{
		{
			name:     "nil input",
			input:    nil,
			expected: "",
		},
		{
			name:     "string input",
			input:    "5m",
			expected: "5m",
		},
		{
			name:     "empty string input",
			input:    "",
			expected: "",
		},
		{
			name:     "int input",
			input:    300,
			expected: "300s",
		},
		{
			name:     "float input",
			input:    300.5,
			expected: "301s",
		},
		{
			name:     "json.Number int input",
			input:    json.Number("300"),
			expected: "300s",
		},
		{
			name:     "json.Number float input",
			input:    json.Number("300.5"),
			expected: "301s",
		},
		{
			name:     "invalid json.Number",
			input:    json.Number("invalid"),
			expected: "",
		},
		{
			name:     "bool input (fallback)",
			input:    true,
			expected: "true",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := normalizeKeepAlive(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestOllamaChatHandlerWithKeepAlive tests that the chat handler properly processes keep_alive field
func TestOllamaChatHandlerWithKeepAlive(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name           string
		requestBody    string
		expectedStatus int
		shouldError    bool
	}{
		{
			name: "string keep_alive",
			requestBody: `{
				"model": "test-model",
				"messages": [{"role": "user", "content": "hello"}],
				"keep_alive": "5m"
			}`,
			expectedStatus: http.StatusInternalServerError, // Expected because model won't exist
			shouldError:    false,
		},
		{
			name: "numeric keep_alive",
			requestBody: `{
				"model": "test-model",
				"messages": [{"role": "user", "content": "hello"}],
				"keep_alive": 300
			}`,
			expectedStatus: http.StatusInternalServerError, // Expected because model won't exist
			shouldError:    false,
		},
		{
			name: "float keep_alive",
			requestBody: `{
				"model": "test-model",
				"messages": [{"role": "user", "content": "hello"}],
				"keep_alive": 300.5
			}`,
			expectedStatus: http.StatusInternalServerError, // Expected because model won't exist
			shouldError:    false,
		},
		{
			name: "null keep_alive",
			requestBody: `{
				"model": "test-model",
				"messages": [{"role": "user", "content": "hello"}],
				"keep_alive": null
			}`,
			expectedStatus: http.StatusInternalServerError, // Expected because model won't exist
			shouldError:    false,
		},
		{
			name: "missing keep_alive",
			requestBody: `{
				"model": "test-model",
				"messages": [{"role": "user", "content": "hello"}]
			}`,
			expectedStatus: http.StatusInternalServerError, // Expected because model won't exist
			shouldError:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pm := &ProxyManager{}

			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request, _ = http.NewRequest("POST", "/api/chat", bytes.NewBufferString(tt.requestBody))
			c.Request.Header.Set("Content-Type", "application/json")

			handler := pm.ollamaChatHandler()
			handler(c)

			// If we expect JSON unmarshaling to fail, check for 400
			if tt.shouldError {
				assert.Equal(t, http.StatusBadRequest, w.Code)
			} else {
				// We expect 500 here because the model doesn't exist, but 400 would indicate JSON parsing failed
				assert.NotEqual(t, http.StatusBadRequest, w.Code)
			}
		})
	}
}

// TestOllamaGenerateHandlerWithKeepAlive tests that the generate handler properly processes keep_alive field
func TestOllamaGenerateHandlerWithKeepAlive(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name           string
		requestBody    string
		expectedStatus int
		shouldError    bool
	}{
		{
			name: "string keep_alive",
			requestBody: `{
				"model": "test-model",
				"prompt": "hello world",
				"keep_alive": "5m"
			}`,
			expectedStatus: http.StatusInternalServerError, // Expected because model won't exist
			shouldError:    false,
		},
		{
			name: "numeric keep_alive",
			requestBody: `{
				"model": "test-model",
				"prompt": "hello world",
				"keep_alive": 300
			}`,
			expectedStatus: http.StatusInternalServerError, // Expected because model won't exist
			shouldError:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pm := &ProxyManager{}

			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request, _ = http.NewRequest("POST", "/api/generate", bytes.NewBufferString(tt.requestBody))
			c.Request.Header.Set("Content-Type", "application/json")

			handler := pm.ollamaGenerateHandler()
			handler(c)

			// If we expect JSON unmarshaling to fail, check for 400
			if tt.shouldError {
				assert.Equal(t, http.StatusBadRequest, w.Code)
			} else {
				// We expect 500 here because the model doesn't exist, but 400 would indicate JSON parsing failed
				assert.NotEqual(t, http.StatusBadRequest, w.Code)
			}
		})
	}
}

// TestOllamaEmbedHandlerWithKeepAlive tests that the embed handler properly processes keep_alive field
func TestOllamaEmbedHandlerWithKeepAlive(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name           string
		requestBody    string
		expectedStatus int
		shouldError    bool
	}{
		{
			name: "string keep_alive",
			requestBody: `{
				"model": "test-model",
				"input": "hello world",
				"keep_alive": "5m"
			}`,
			expectedStatus: http.StatusInternalServerError, // Expected because model won't exist
			shouldError:    false,
		},
		{
			name: "numeric keep_alive",
			requestBody: `{
				"model": "test-model",
				"input": "hello world",
				"keep_alive": 300
			}`,
			expectedStatus: http.StatusInternalServerError, // Expected because model won't exist
			shouldError:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pm := &ProxyManager{}

			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request, _ = http.NewRequest("POST", "/api/embed", bytes.NewBufferString(tt.requestBody))
			c.Request.Header.Set("Content-Type", "application/json")

			handler := pm.ollamaEmbedHandler()
			handler(c)

			// If we expect JSON unmarshaling to fail, check for 400
			if tt.shouldError {
				assert.Equal(t, http.StatusBadRequest, w.Code)
			} else {
				// We expect 500 here because the model doesn't exist, but 400 would indicate JSON parsing failed
				assert.NotEqual(t, http.StatusBadRequest, w.Code)
			}
		})
	}
}

// TestOllamaLegacyEmbeddingsHandlerWithKeepAlive tests that the legacy embeddings handler properly processes keep_alive field
func TestOllamaLegacyEmbeddingsHandlerWithKeepAlive(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name           string
		requestBody    string
		expectedStatus int
		shouldError    bool
	}{
		{
			name: "string keep_alive",
			requestBody: `{
				"model": "test-model",
				"prompt": "hello world",
				"keep_alive": "5m"
			}`,
			expectedStatus: http.StatusInternalServerError, // Expected because model won't exist
			shouldError:    false,
		},
		{
			name: "numeric keep_alive",
			requestBody: `{
				"model": "test-model",
				"prompt": "hello world",
				"keep_alive": 300
			}`,
			expectedStatus: http.StatusInternalServerError, // Expected because model won't exist
			shouldError:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pm := &ProxyManager{}

			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request, _ = http.NewRequest("POST", "/api/embeddings", bytes.NewBufferString(tt.requestBody))
			c.Request.Header.Set("Content-Type", "application/json")

			handler := pm.ollamaLegacyEmbeddingsHandler()
			handler(c)

			// If we expect JSON unmarshaling to fail, check for 400
			if tt.shouldError {
				assert.Equal(t, http.StatusBadRequest, w.Code)
			} else {
				// We expect 500 here because the model doesn't exist, but 400 would indicate JSON parsing failed
				assert.NotEqual(t, http.StatusBadRequest, w.Code)
			}
		})
	}
}

// TestOllamaRequestStructs tests that the request structs can be properly unmarshaled with different keep_alive types
func TestOllamaRequestStructs(t *testing.T) {
	tests := []struct {
		name        string
		requestJSON string
		structType  string
	}{
		{
			name: "ChatRequest with numeric keep_alive",
			requestJSON: `{
				"model": "test-model",
				"messages": [{"role": "user", "content": "hello"}],
				"keep_alive": 300
			}`,
			structType: "OllamaChatRequest",
		},
		{
			name: "ChatRequest with string keep_alive",
			requestJSON: `{
				"model": "test-model",
				"messages": [{"role": "user", "content": "hello"}],
				"keep_alive": "5m"
			}`,
			structType: "OllamaChatRequest",
		},
		{
			name: "GenerateRequest with numeric keep_alive",
			requestJSON: `{
				"model": "test-model",
				"prompt": "hello",
				"keep_alive": 300
			}`,
			structType: "OllamaGenerateRequest",
		},
		{
			name: "EmbedRequest with numeric keep_alive",
			requestJSON: `{
				"model": "test-model",
				"input": "hello",
				"keep_alive": 300
			}`,
			structType: "OllamaEmbedRequest",
		},
		{
			name: "LegacyEmbeddingsRequest with numeric keep_alive",
			requestJSON: `{
				"model": "test-model",
				"prompt": "hello",
				"keep_alive": 300
			}`,
			structType: "OllamaLegacyEmbeddingsRequest",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			switch tt.structType {
			case "OllamaChatRequest":
				var req OllamaChatRequest
				err = json.Unmarshal([]byte(tt.requestJSON), &req)
				assert.NoError(t, err)
				assert.Equal(t, "test-model", req.Model)
				if tt.name == "ChatRequest with string keep_alive" {
					assert.Equal(t, "5m", req.KeepAlive)
				} else {
					assert.Equal(t, 300.0, req.KeepAlive)
				}

			case "OllamaGenerateRequest":
				var req OllamaGenerateRequest
				err = json.Unmarshal([]byte(tt.requestJSON), &req)
				assert.NoError(t, err)
				assert.Equal(t, "test-model", req.Model)
				assert.Equal(t, 300.0, req.KeepAlive)

			case "OllamaEmbedRequest":
				var req OllamaEmbedRequest
				err = json.Unmarshal([]byte(tt.requestJSON), &req)
				assert.NoError(t, err)
				assert.Equal(t, "test-model", req.Model)
				assert.Equal(t, 300.0, req.KeepAlive)

			case "OllamaLegacyEmbeddingsRequest":
				var req OllamaLegacyEmbeddingsRequest
				err = json.Unmarshal([]byte(tt.requestJSON), &req)
				assert.NoError(t, err)
				assert.Equal(t, "test-model", req.Model)
				assert.Equal(t, 300.0, req.KeepAlive)
			}

			// The main test is that JSON unmarshaling doesn't fail with numeric keep_alive
			assert.NoError(t, err)
		})
	}
}

// TestCreateOpenAIRequestBodyWithThink tests that the think parameter is correctly translated
func TestCreateOpenAIRequestBodyWithThink(t *testing.T) {
	tests := []struct {
		name           string
		think          *bool
		format         interface{}
		expectKwargs   bool
		expectThinking *bool
		expectFormat   bool
	}{
		{
			name:           "think=true",
			think:          boolPtr(true),
			expectKwargs:   true,
			expectThinking: boolPtr(true),
		},
		{
			name:           "think=false",
			think:          boolPtr(false),
			expectKwargs:   true,
			expectThinking: boolPtr(false),
		},
		{
			name:         "think=nil (no param)",
			think:        nil,
			expectKwargs: false,
		},
		{
			name:         "format=json",
			format:       "json",
			expectFormat: true,
		},
		{
			name: "format=json_schema",
			format: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]interface{}{"type": "string"},
				},
			},
			expectFormat: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := &createOpenAIRequestBodyOptions{
				Think:  tt.think,
				Format: tt.format,
			}
			messages := []map[string]interface{}{
				{"role": "user", "content": "test"},
			}

			bodyBytes, err := createOpenAIRequestBody("test-model", messages, false, nil, nil, nil, opts)
			assert.NoError(t, err)

			var result map[string]interface{}
			err = json.Unmarshal(bodyBytes, &result)
			assert.NoError(t, err)

			if tt.expectKwargs {
				kwargs, ok := result["chat_template_kwargs"].(map[string]interface{})
				assert.True(t, ok, "chat_template_kwargs should exist")
				enableThinking, ok := kwargs["enable_thinking"].(bool)
				assert.True(t, ok, "enable_thinking should be a bool")
				assert.Equal(t, *tt.expectThinking, enableThinking)
			} else {
				_, ok := result["chat_template_kwargs"]
				assert.False(t, ok, "chat_template_kwargs should not exist")
			}

			if tt.expectFormat {
				_, ok := result["response_format"]
				assert.True(t, ok, "response_format should exist")
			}
		})
	}
}

// TestReasoningContentToThinking tests that OpenAI reasoning_content is mapped to Ollama thinking field
func TestReasoningContentToThinking(t *testing.T) {
	tests := []struct {
		name                    string
		openAIResponse          string
		expectedThinking        string
		expectedContent         string
		shouldHaveThinkingField bool
	}{
		{
			name: "response with reasoning_content",
			openAIResponse: `{
				"id": "test",
				"object": "chat.completion",
				"created": 1234567890,
				"model": "test-model",
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "The answer is 4.",
						"reasoning_content": "Let me think about this... 2 + 2 = 4."
					},
					"finish_reason": "stop"
				}],
				"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
			}`,
			expectedThinking:        "Let me think about this... 2 + 2 = 4.",
			expectedContent:         "The answer is 4.",
			shouldHaveThinkingField: true,
		},
		{
			name: "response without reasoning_content",
			openAIResponse: `{
				"id": "test",
				"object": "chat.completion",
				"created": 1234567890,
				"model": "test-model",
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "The answer is 4."
					},
					"finish_reason": "stop"
				}],
				"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
			}`,
			expectedThinking:        "",
			expectedContent:         "The answer is 4.",
			shouldHaveThinkingField: false,
		},
		{
			name: "response with empty reasoning_content",
			openAIResponse: `{
				"id": "test",
				"object": "chat.completion",
				"created": 1234567890,
				"model": "test-model",
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "Quick answer.",
						"reasoning_content": ""
					},
					"finish_reason": "stop"
				}],
				"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
			}`,
			expectedThinking:        "",
			expectedContent:         "Quick answer.",
			shouldHaveThinkingField: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var openAIResp OpenAIChatCompletionResponse
			err := json.Unmarshal([]byte(tt.openAIResponse), &openAIResp)
			assert.NoError(t, err)
			assert.Len(t, openAIResp.Choices, 1)

			choice := openAIResp.Choices[0]
			message := OllamaMessage{
				Role:     openAIRoleToOllama(choice.Message.Role),
				Content:  choice.Message.Content,
				Thinking: choice.Message.ReasoningContent,
			}

			assert.Equal(t, tt.expectedContent, message.Content)
			assert.Equal(t, tt.expectedThinking, message.Thinking)

			// Verify JSON serialization only includes thinking field when non-empty
			jsonBytes, err := json.Marshal(message)
			assert.NoError(t, err)

			var result map[string]interface{}
			err = json.Unmarshal(jsonBytes, &result)
			assert.NoError(t, err)

			if tt.shouldHaveThinkingField {
				thinking, ok := result["thinking"]
				assert.True(t, ok, "thinking field should be present in JSON")
				assert.Equal(t, tt.expectedThinking, thinking)
			} else {
				_, ok := result["thinking"]
				assert.False(t, ok, "thinking field should be omitted when empty")
			}
		})
	}
}

// TestOllamaChatHandlerSendsChatTemplateKwargs verifies that when think=true is sent
// via the Ollama API, the request forwarded to the backend includes chat_template_kwargs
func TestOllamaChatHandlerSendsChatTemplateKwargs(t *testing.T) {
	// This test reproduces the bug where think=true in Ollama API doesn't result in
	// chat_template_kwargs being sent to the llama.cpp backend

	// Parse an Ollama request with think=true
	ollamaRequestJSON := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "Hello"}],
		"think": true,
		"stream": false
	}`

	var ollamaReq OllamaChatRequest
	err := json.Unmarshal([]byte(ollamaRequestJSON), &ollamaReq)
	assert.NoError(t, err)
	assert.NotNil(t, ollamaReq.Think, "Think should be parsed")
	assert.True(t, *ollamaReq.Think, "Think should be true")

	// Simulate what ollamaChatHandler does: create the OpenAI request body
	openAIMessages := ollamaMessagesToOpenAI(ollamaReq.Messages)
	openAITools := ollamaToolsToOpenAI(ollamaReq.Tools)

	isStreaming := ollamaReq.Stream != nil && *ollamaReq.Stream
	opts := &createOpenAIRequestBodyOptions{
		Think:  ollamaReq.Think,
		Format: ollamaReq.Format,
	}

	openAIReqBodyBytes, err := createOpenAIRequestBody(
		"test-model",
		openAIMessages,
		isStreaming,
		ollamaReq.Options,
		openAITools,
		ollamaReq.ToolChoice,
		opts,
	)
	assert.NoError(t, err)

	// Verify the request body contains chat_template_kwargs with enable_thinking=true
	var requestBody map[string]interface{}
	err = json.Unmarshal(openAIReqBodyBytes, &requestBody)
	assert.NoError(t, err)

	chatTemplateKwargs, ok := requestBody["chat_template_kwargs"].(map[string]interface{})
	assert.True(t, ok, "chat_template_kwargs should exist in request body")

	enableThinking, ok := chatTemplateKwargs["enable_thinking"].(bool)
	assert.True(t, ok, "enable_thinking should be a bool")
	assert.True(t, enableThinking, "enable_thinking should be true")
}

// TestOllamaChatRequestWithThink tests parsing OllamaChatRequest with think parameter
func TestOllamaChatRequestWithThink(t *testing.T) {
	tests := []struct {
		name        string
		requestJSON string
		expectThink *bool
	}{
		{
			name:        "think=true",
			requestJSON: `{"model": "test-model", "messages": [], "think": true}`,
			expectThink: boolPtr(true),
		},
		{
			name:        "think=false",
			requestJSON: `{"model": "test-model", "messages": [], "think": false}`,
			expectThink: boolPtr(false),
		},
		{
			name:        "no think parameter",
			requestJSON: `{"model": "test-model", "messages": []}`,
			expectThink: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req OllamaChatRequest
			err := json.Unmarshal([]byte(tt.requestJSON), &req)
			assert.NoError(t, err)
			assert.Equal(t, "test-model", req.Model)

			if tt.expectThink == nil {
				assert.Nil(t, req.Think)
			} else {
				assert.NotNil(t, req.Think)
				assert.Equal(t, *tt.expectThink, *req.Think)
			}
		})
	}
}
