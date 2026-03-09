package server

import (
	"fmt"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ollamaSpoofedVersion is what llama-swappo reports as its version. Ollama
// clients gate features on the version string, so llama-swap's own version
// would make them fall back to a much smaller API surface.
const ollamaSpoofedVersion = "0.13.5"

// applyOllamaCompat rewrites an OpenAI-format request body with the
// compatibility fixups llama-swappo adds on top of llama-swap. It runs after
// the standard filters, on the same buffered body, and returns the body
// unchanged when none of the fixups apply.
func applyOllamaCompat(body []byte, mc config.ModelConfig) ([]byte, error) {
	body, err := ensureOpenAIToolParameters(body)
	if err != nil {
		return nil, err
	}
	body, err = applyChatTemplateKwargs(body, mc)
	if err != nil {
		return nil, err
	}
	return translateThinkParam(body)
}

// ensureOpenAIToolParameters adds an empty parameters object to any function
// tool that omits one. Some clients (Copilot) leave the field out entirely for
// a no-argument function; llama-server rejects the schema when they do.
func ensureOpenAIToolParameters(body []byte) ([]byte, error) {
	tools := gjson.GetBytes(body, "tools")
	if !tools.Exists() || !tools.IsArray() {
		return body, nil
	}

	updated := body
	for i, tool := range tools.Array() {
		if tool.Get("type").String() != "function" {
			continue
		}
		functionPath := fmt.Sprintf("tools.%d.function", i)
		function := gjson.GetBytes(updated, functionPath)
		if !function.Exists() || !function.IsObject() {
			continue
		}
		parametersPath := functionPath + ".parameters"
		if gjson.GetBytes(updated, parametersPath).Exists() {
			continue
		}

		next, err := sjson.SetBytes(updated, parametersPath, map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		})
		if err != nil {
			return nil, fmt.Errorf("failed to set default parameters for tool %d: %w", i, err)
		}
		updated = next
	}
	return updated, nil
}

// applyChatTemplateKwargs merges the model's configured chatTemplateKwargs into
// the request. Request-level values win, so a config default only fills a key
// the client left unset.
func applyChatTemplateKwargs(body []byte, mc config.ModelConfig) ([]byte, error) {
	if len(mc.ChatTemplateKwargs) == 0 {
		return body, nil
	}

	if !gjson.GetBytes(body, "chat_template_kwargs").Exists() {
		updated, err := sjson.SetBytes(body, "chat_template_kwargs", mc.ChatTemplateKwargs)
		if err != nil {
			return nil, fmt.Errorf("error setting chat_template_kwargs from config: %w", err)
		}
		return updated, nil
	}

	updated := body
	for key, value := range mc.ChatTemplateKwargs {
		path := "chat_template_kwargs." + key
		if gjson.GetBytes(updated, path).Exists() {
			continue
		}
		next, err := sjson.SetBytes(updated, path, value)
		if err != nil {
			return nil, fmt.Errorf("error merging chat_template_kwargs.%s: %w", key, err)
		}
		updated = next
	}
	return updated, nil
}

// translateThinkParam rewrites Ollama's "think" into llama-server's
// chat_template_kwargs.enable_thinking, letting OpenAI clients drive thinking
// mode with the Ollama convention.
func translateThinkParam(body []byte) ([]byte, error) {
	think := gjson.GetBytes(body, "think")
	if !think.Exists() {
		return body, nil
	}

	updated, err := sjson.SetBytes(body, "chat_template_kwargs.enable_thinking", think.Bool())
	if err != nil {
		return nil, fmt.Errorf("error setting enable_thinking: %w", err)
	}
	updated, err = sjson.DeleteBytes(updated, "think")
	if err != nil {
		return nil, fmt.Errorf("error removing think from request: %w", err)
	}
	return updated, nil
}
