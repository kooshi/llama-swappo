package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServer_OllamaCompatEnsureToolParameters(t *testing.T) {
	tests := []struct {
		name string
		body string
		want map[string]any
	}{
		{
			name: "function tool missing parameters gets an empty object",
			body: `{"tools":[{"type":"function","function":{"name":"now"}}]}`,
			want: map[string]any{"type": "object", "properties": map[string]any{}},
		},
		{
			name: "existing parameters are left alone",
			body: `{"tools":[{"type":"function","function":{"name":"now","parameters":{"type":"object","properties":{"tz":{"type":"string"}}}}}]}`,
			want: map[string]any{"type": "object", "properties": map[string]any{"tz": map[string]any{"type": "string"}}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := applyOllamaCompat([]byte(tt.body), config.ModelConfig{})
			require.NoError(t, err)

			var parsed map[string]any
			require.NoError(t, json.Unmarshal(got, &parsed))

			tools := parsed["tools"].([]any)
			fn := tools[0].(map[string]any)["function"].(map[string]any)
			assert.Equal(t, tt.want, fn["parameters"])
		})
	}
}

func TestServer_OllamaCompatNonFunctionToolsUntouched(t *testing.T) {
	body := `{"tools":[{"type":"custom","function":{"name":"now"}}]}`
	got, err := applyOllamaCompat([]byte(body), config.ModelConfig{})
	require.NoError(t, err)
	assert.JSONEq(t, body, string(got))
}

func TestServer_OllamaCompatChatTemplateKwargs(t *testing.T) {
	mc := config.ModelConfig{ChatTemplateKwargs: map[string]any{"enable_thinking": true, "tools_in_user_message": false}}

	tests := []struct {
		name string
		body string
		want map[string]any
	}{
		{
			name: "config defaults are applied when the request has none",
			body: `{"model":"m"}`,
			want: map[string]any{"enable_thinking": true, "tools_in_user_message": false},
		},
		{
			name: "request values win over config defaults",
			body: `{"model":"m","chat_template_kwargs":{"enable_thinking":false}}`,
			want: map[string]any{"enable_thinking": false, "tools_in_user_message": false},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := applyOllamaCompat([]byte(tt.body), mc)
			require.NoError(t, err)

			var parsed map[string]any
			require.NoError(t, json.Unmarshal(got, &parsed))
			assert.Equal(t, tt.want, parsed["chat_template_kwargs"])
		})
	}
}

func TestServer_OllamaCompatTranslatesThink(t *testing.T) {
	tests := []struct {
		name string
		body string
		want any
	}{
		{name: "think true", body: `{"model":"m","think":true}`, want: true},
		{name: "think false", body: `{"model":"m","think":false}`, want: false},
		{
			name: "think overrides an existing enable_thinking",
			body: `{"model":"m","think":true,"chat_template_kwargs":{"enable_thinking":false}}`,
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := applyOllamaCompat([]byte(tt.body), config.ModelConfig{})
			require.NoError(t, err)

			var parsed map[string]any
			require.NoError(t, json.Unmarshal(got, &parsed))

			_, stillThere := parsed["think"]
			assert.False(t, stillThere, "think should be consumed, not forwarded upstream")
			assert.Equal(t, tt.want, parsed["chat_template_kwargs"].(map[string]any)["enable_thinking"])
		})
	}
}

func TestServer_OllamaCompatLeavesPlainRequestsAlone(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
	got, err := applyOllamaCompat([]byte(body), config.ModelConfig{})
	require.NoError(t, err)
	assert.JSONEq(t, body, string(got))
}

// TestServer_OllamaRoutes covers the routes that share a path with a llama-swap
// endpoint: registering them for HEAD only must leave the GET handler serving.
func TestServer_OllamaRoutes(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))
	s.build = BuildInfo{Version: "1.2.3"}

	t.Run("HEAD / is the Ollama heartbeat", func(t *testing.T) {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodHead, "/", nil))
		assert.Equal(t, http.StatusOK, w.Code)
	})

	t.Run("GET / still redirects to the UI", func(t *testing.T) {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/", nil))
		assert.Equal(t, http.StatusFound, w.Code)
		assert.Equal(t, "/ui", w.Header().Get("Location"))
	})

	t.Run("HEAD /api/version answers", func(t *testing.T) {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodHead, "/api/version", nil))
		assert.Equal(t, http.StatusOK, w.Code)
	})

	t.Run("GET /api/tags lists models", func(t *testing.T) {
		s.cfg = config.Config{Models: map[string]config.ModelConfig{
			"qwen3-32b": {Cmd: "llama-server"},
			"hidden":    {Cmd: "llama-server", Unlisted: true},
		}}

		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/tags", nil))
		require.Equal(t, http.StatusOK, w.Code)

		var resp OllamaListTagsResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		require.Len(t, resp.Models, 1)
		assert.Equal(t, "qwen3-32b", resp.Models[0].Name)
	})

	t.Run("stubbed endpoints report not implemented", func(t *testing.T) {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/pull", nil))
		assert.Equal(t, http.StatusNotImplemented, w.Code)
	})
}

// TestServer_OllamaStreamingUpstreamError checks that an error raised while a
// streaming request is being dispatched reaches the client. The transform only
// understands SSE, so an error body has to bypass it.
func TestServer_OllamaStreamingUpstreamError(t *testing.T) {
	cfg := config.Config{Models: map[string]config.ModelConfig{
		"test-model": {Cmd: "llama-server", CheckEndpoint: "none"},
	}}
	cfg = config.AddDefaultGroupToConfig(cfg)

	s := newOllamaTestServer(t, cfg, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"upstream exploded"}}`, http.StatusBadGateway)
	})

	body := `{"model":"test-model","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	s.ollamaChatHandler()(w, req)

	assert.Equal(t, http.StatusBadGateway, w.Code)
	assert.Contains(t, w.Body.String(), "upstream exploded")
}
