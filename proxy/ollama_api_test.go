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