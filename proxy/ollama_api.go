package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// normalizeKeepAlive converts keep_alive from interface{} to string format
// Accepts numbers (int, float64) and converts to duration string, or passes through strings
func normalizeKeepAlive(keepAlive interface{}) string {
	if keepAlive == nil {
		return ""
	}

	switch v := keepAlive.(type) {
	case string:
		return v
	case int:
		// Convert seconds to duration string (e.g., 300 -> "5m")
		return fmt.Sprintf("%ds", v)
	case float64:
		// Convert seconds to duration string, handling fractional seconds with proper rounding
		return fmt.Sprintf("%.0fs", v+0.5)
	case json.Number:
		// Handle JSON numbers that come as string
		if num, err := v.Int64(); err == nil {
			return fmt.Sprintf("%ds", num)
		}
		// Try float if int conversion fails
		if num, err := v.Float64(); err == nil {
			return fmt.Sprintf("%.0fs", num+0.5)
		}
		return ""
	default:
		// For any other type, try to convert to string
		return fmt.Sprintf("%v", v)
	}
}

func (pm *ProxyManager) sendOllamaError(c *gin.Context, statusCode int, message string) {
	c.JSON(statusCode, OllamaErrorResponse{Error: message})
}

func (pm *ProxyManager) ollamaNotImplementedHandler(c *gin.Context) {
	pm.sendOllamaError(c, http.StatusNotImplemented, "This Ollama API endpoint is not implemented in llama-swap.")
}

func (pm *ProxyManager) ollamaVersionHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Handle CORS if Origin header is present
		if origin := c.Request.Header.Get("Origin"); origin != "" {
			c.Header("Access-Control-Allow-Origin", origin)
		}
		c.JSON(http.StatusOK, OllamaVersionResponse{Version: "0.0.0"})
	}
}

func (pm *ProxyManager) ollamaHeartbeatHandler(c *gin.Context) {
	c.String(http.StatusOK, "Ollama is running") // Ollama server returns this string
}

func (pm *ProxyManager) ollamaListTagsHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		models := []OllamaModelResponse{}
		now := time.Now().UTC() // Use a consistent timestamp

		pm.RLock() // Lock for reading pm.config.Models
		for id, modelCfg := range pm.config.Models {
			if modelCfg.Unlisted {
				continue
			}
			details := OllamaModelDetails{Format: "gguf"}
			if family, ok := modelCfg.Metadata["family"].(string); ok && family != "" {
				details.Family = family
			} else {
				// Basic inference for list view
				arch := "unknown"
				if v, ok := modelCfg.Metadata["architecture"].(string); ok && v != "" {
					arch = v
				} else {
					arch = inferPattern(id, architecturePatterns, orderedArchKeys)
				}
				details.Family = inferFamilyFromName(id, arch)
			}
			if paramSize, ok := modelCfg.Metadata["parameterSize"].(string); ok && paramSize != "" {
				details.ParameterSize = paramSize
			} else {
				details.ParameterSize = inferParameterSizeFromName(id)
			}
			if quantLevel, ok := modelCfg.Metadata["quantizationLevel"].(string); ok && quantLevel != "" {
				details.QuantizationLevel = quantLevel
			} else {
				details.QuantizationLevel = inferQuantizationLevelFromName(id)
			}
			if details.Family != "unknown" && details.Family != "" {
				details.Families = []string{details.Family}
			}

			models = append(models, OllamaModelResponse{
				Name:       id,
				Model:      id,
				ModifiedAt: now,
				Size:       0,
				Digest:     fmt.Sprintf("%x", id),
				Details:    details,
			})
		}
		pm.RUnlock()

		// Handle CORS if Origin header is present
		if origin := c.Request.Header.Get("Origin"); origin != "" {
			c.Header("Access-Control-Allow-Origin", origin)
		}

		c.JSON(http.StatusOK, OllamaListTagsResponse{Models: models})
	}
}

func (pm *ProxyManager) ollamaShowHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req OllamaShowRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			pm.sendOllamaError(c, http.StatusBadRequest, fmt.Sprintf("Invalid request: %v", err))
			return
		}

		modelName := req.Model
		if modelName == "" {
			modelName = req.Name
		}

		if modelName == "" {
			pm.sendOllamaError(c, http.StatusBadRequest, "Model name is required.")
			return
		}

		pm.RLock()
		modelCfg, id, found := pm.config.FindConfig(modelName) // id is realModelName
		pm.RUnlock()

		if !found {
			pm.sendOllamaError(c, http.StatusNotFound, fmt.Sprintf("Model '%s' not found.", modelName))
			return
		}

		parser := NewLlamaServerParser()
		parsedArgs := parser.Parse(modelCfg.Cmd, id)

		arch := parsedArgs.Architecture
		family := parsedArgs.Family
		paramSize := parsedArgs.ParameterSize
		quantLevel := parsedArgs.QuantizationLevel
		ctxLength := parsedArgs.ContextLength
		caps := parsedArgs.Capabilities
		if len(caps) == 0 {
			caps = []string{"completion"}
		}

		// Override with metadata if present
		if v, ok := modelCfg.Metadata["architecture"].(string); ok && v != "" {
			arch = v
		}
		if v, ok := modelCfg.Metadata["family"].(string); ok && v != "" {
			family = v
		}
		if v, ok := modelCfg.Metadata["parameterSize"].(string); ok && v != "" {
			paramSize = v
		}
		if v, ok := modelCfg.Metadata["quantizationLevel"].(string); ok && v != "" {
			quantLevel = v
		}
		if v, ok := modelCfg.Metadata["contextLength"].(int); ok && v != 0 {
			ctxLength = v
		}
		if v, ok := modelCfg.Metadata["capabilities"].([]any); ok && len(v) > 0 {
			newCaps := make([]string, 0, len(v))
			for _, item := range v {
				if s, isString := item.(string); isString {
					newCaps = append(newCaps, s)
				}
			}
			if len(newCaps) > 0 {
				caps = newCaps
			}
		}

		details := OllamaModelDetails{
			Format:            "gguf",
			Family:            family,
			ParameterSize:     paramSize,
			QuantizationLevel: quantLevel,
		}
		if family != "unknown" && family != "" {
			details.Families = []string{family}
		}

		modelInfo := map[string]interface{}{
			"general.architecture": arch,
		}
		if ctxLength > 0 {
			modelInfo["llama.context_length"] = ctxLength
		} else {
			modelInfo["llama.context_length"] = 2048
		}

		resp := OllamaShowResponse{
			Details:      details,
			ModelInfo:    modelInfo,
			Capabilities: caps,
		}

		// Handle CORS if Origin header is present
		if origin := c.Request.Header.Get("Origin"); origin != "" {
			c.Header("Access-Control-Allow-Origin", origin)
		}

		c.JSON(http.StatusOK, resp)
	}
}

func (pm *ProxyManager) ollamaPSHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		runningModels := []OllamaProcessModelResponse{}
		now := time.Now().UTC()

		pm.RLock()
		for _, group := range pm.processGroups {
			group.Lock() // Lock group while iterating its processes
			for modelID, process := range group.processes {
				if process.CurrentState() == StateReady {
					expiresAt := time.Time{} // Zero time if no TTL
					if process.config.UnloadAfter > 0 {
						expiresAt = process.lastRequestHandled.Add(time.Duration(process.config.UnloadAfter) * time.Second)
						if expiresAt.Before(now) && !process.lastRequestHandled.IsZero() {
							expiresAt = now.Add(time.Duration(process.config.UnloadAfter) * time.Second)
						} else if process.lastRequestHandled.IsZero() {
							expiresAt = now.Add(time.Duration(process.config.UnloadAfter) * time.Second)
						}
					}

					modelCfg := process.config
					details := OllamaModelDetails{Format: "gguf"}

					arch := "unknown"
					if v, ok := modelCfg.Metadata["architecture"].(string); ok && v != "" {
						arch = v
					} else {
						arch = inferPattern(modelID, architecturePatterns, orderedArchKeys)
					}

					if v, ok := modelCfg.Metadata["family"].(string); ok && v != "" {
						details.Family = v
					} else {
						details.Family = inferFamilyFromName(modelID, arch)
					}
					if v, ok := modelCfg.Metadata["parameterSize"].(string); ok && v != "" {
						details.ParameterSize = v
					} else {
						details.ParameterSize = inferParameterSizeFromName(modelID)
					}
					if v, ok := modelCfg.Metadata["quantizationLevel"].(string); ok && v != "" {
						details.QuantizationLevel = v
					} else {
						details.QuantizationLevel = inferQuantizationLevelFromName(modelID)
					}
					if details.Family != "unknown" && details.Family != "" {
						details.Families = []string{details.Family}
					}

					runningModels = append(runningModels, OllamaProcessModelResponse{
						Name:      modelID,
						Model:     modelID,
						Size:      0,
						Digest:    fmt.Sprintf("%x", modelID),
						Details:   details,
						ExpiresAt: expiresAt,
						SizeVRAM:  0,
					})
				}
			}
			group.Unlock()
		}
		pm.RUnlock()

		// Handle CORS if Origin header is present
		if origin := c.Request.Header.Get("Origin"); origin != "" {
			c.Header("Access-Control-Allow-Origin", origin)
		}

		c.JSON(http.StatusOK, OllamaProcessResponse{Models: runningModels})
	}
}

// transformingResponseWriter captures and transforms SSE stream from OpenAI to Ollama format
type transformingResponseWriter struct {
	ginWriter      gin.ResponseWriter
	modelName      string
	buffer         bytes.Buffer                 // To handle partial SSE events
	isChat         bool                         // True for chat, false for generate
	toolCallBuffer map[int]*accumulatedToolCall // Accumulate streaming tool call deltas by index
}

// accumulatedToolCall collects streaming tool call deltas until complete
type accumulatedToolCall struct {
	ID        string
	Type      string
	Name      string
	Arguments strings.Builder // accumulate argument fragments
}

func newTransformingResponseWriter(writer gin.ResponseWriter, modelName string, isChat bool) *transformingResponseWriter {
	return &transformingResponseWriter{
		ginWriter:      writer,
		modelName:      modelName,
		isChat:         isChat,
		toolCallBuffer: make(map[int]*accumulatedToolCall),
	}
}

func (trw *transformingResponseWriter) Header() http.Header {
	return trw.ginWriter.Header()
}

func (trw *transformingResponseWriter) Write(data []byte) (int, error) {
	// Append data to internal buffer
	return trw.buffer.Write(data)
}

func (trw *transformingResponseWriter) WriteHeader(statusCode int) {
	trw.ginWriter.WriteHeader(statusCode)
}

func (trw *transformingResponseWriter) Flush() {
	scanner := bufio.NewScanner(&trw.buffer)
	var processedBuffer bytes.Buffer // Store fully processed lines to write

	var unprocessedSuffix []byte // Store any partial line at the end

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			jsonData := strings.TrimPrefix(line, "data: ")
			if jsonData == "[DONE]" {
				break
			}

			var ollamaChunkJSON []byte
			var err error

			if trw.isChat {
				var openAIChatChunk OpenAIStreamingChatResponse
				if err = json.Unmarshal([]byte(jsonData), &openAIChatChunk); err == nil {
					if len(openAIChatChunk.Choices) > 0 {
						choice := openAIChatChunk.Choices[0]
						message := OllamaMessage{
							Role:     openAIRoleToOllama(choice.Delta.Role),
							Content:  choice.Delta.Content,
							Thinking: choice.Delta.ReasoningContent,
						}

						// Handle tool calls in streaming response - ACCUMULATE instead of immediate output
						// OpenAI streams tool calls incrementally (ID, name, arguments fragments)
						// We collect them until the stream ends, then emit complete tool calls
						if len(choice.Delta.ToolCalls) > 0 {
							for _, tcDelta := range choice.Delta.ToolCalls {
								acc, exists := trw.toolCallBuffer[tcDelta.Index]
								if !exists {
									acc = &accumulatedToolCall{}
									trw.toolCallBuffer[tcDelta.Index] = acc
								}
								// Accumulate fields (only update if non-empty)
								if tcDelta.ID != "" {
									acc.ID = tcDelta.ID
								}
								if tcDelta.Type != "" {
									acc.Type = tcDelta.Type
								}
								if tcDelta.Function.Name != "" {
									acc.Name = tcDelta.Function.Name
								}
								// Arguments come as string fragments - concatenate
								acc.Arguments.WriteString(tcDelta.Function.Arguments)
							}
							// DON'T add tool calls to message yet - emit on final chunk
						}

						// On final chunk, emit accumulated tool calls
						if choice.FinishReason != "" && len(trw.toolCallBuffer) > 0 {
							ollamaToolCalls := make([]OllamaToolCall, 0, len(trw.toolCallBuffer))

							// Sort by index for consistent ordering
							indices := make([]int, 0, len(trw.toolCallBuffer))
							for idx := range trw.toolCallBuffer {
								indices = append(indices, idx)
							}
							sort.Ints(indices)

							for _, idx := range indices {
								acc := trw.toolCallBuffer[idx]
								// Skip invalid tool calls (empty name = hallucinated)
								if acc.Name == "" {
									continue
								}
								var args map[string]interface{}
								if argsStr := acc.Arguments.String(); argsStr != "" {
									json.Unmarshal([]byte(argsStr), &args)
								}
								ollamaToolCalls = append(ollamaToolCalls, OllamaToolCall{
									ID:   acc.ID,
									Type: acc.Type,
									Function: OllamaToolCallFunc{
										Index:     idx,
										Name:      acc.Name,
										Arguments: args,
									},
								})
							}
							if len(ollamaToolCalls) > 0 {
								message.ToolCalls = ollamaToolCalls
							}
						}

						ollamaResp := OllamaChatResponse{
							Model:      trw.modelName,
							CreatedAt:  time.Now().UTC(),
							Message:    message,
							Done:       choice.FinishReason != "",
							DoneReason: openAIFinishReasonToOllama(choice.FinishReason),
						}
						if choice.Delta.Role == "" && ollamaResp.Message.Role == "" {
							ollamaResp.Message.Role = "assistant"
						}
						if openAIChatChunk.Usage != nil {
							ollamaResp.PromptEvalCount = openAIChatChunk.Usage.PromptTokens
							ollamaResp.EvalCount = openAIChatChunk.Usage.CompletionTokens
						}

						ollamaChunkJSON, err = json.Marshal(ollamaResp)
					}
				}
			} else { // /api/generate
				var openAIGenChunk OpenAIStreamingCompletionResponse
				if err = json.Unmarshal([]byte(jsonData), &openAIGenChunk); err == nil {
					if len(openAIGenChunk.Choices) > 0 {
						choice := openAIGenChunk.Choices[0]
						ollamaResp := OllamaGenerateResponse{
							Model:      trw.modelName,
							CreatedAt:  time.Now().UTC(),
							Response:   choice.Text,
							Done:       choice.FinishReason != "",
							DoneReason: openAIFinishReasonToOllama(choice.FinishReason),
						}
						if openAIGenChunk.Usage != nil {
							ollamaResp.PromptEvalCount = openAIGenChunk.Usage.PromptTokens
							ollamaResp.EvalCount = openAIGenChunk.Usage.CompletionTokens
						}
						ollamaChunkJSON, err = json.Marshal(ollamaResp)
					}
				}
			}

			if err == nil && ollamaChunkJSON != nil {
				processedBuffer.Write(ollamaChunkJSON)
				processedBuffer.WriteString("\n")
			} else if err != nil {
				fmt.Fprintf(trw.ginWriter, "{\"error\":\"Error transforming stream: %v\"}\n", err)
			}
		} else if line != "" {
			var errResp OllamaErrorResponse
			if json.Unmarshal([]byte(line), &errResp) == nil && errResp.Error != "" {
				processedBuffer.Write([]byte(line))
				processedBuffer.WriteString("\n")
			}
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(trw.ginWriter, "{\"error\":\"Error scanning stream buffer: %v\"}\n", err)
	}

	// If there is any unprocessed suffix, write it back to the buffer
	unprocessedSuffix = nil
	if trw.buffer.Len() > 0 && len(scanner.Bytes()) > 0 && trw.buffer.Len() >= len(scanner.Bytes()) {
		unprocessedSuffix = trw.buffer.Bytes()[trw.buffer.Len()-len(scanner.Bytes()):]
	}
	trw.buffer.Reset()
	if unprocessedSuffix != nil {
		trw.buffer.Write(unprocessedSuffix)
	}

	if processedBuffer.Len() > 0 {
		trw.ginWriter.Write(processedBuffer.Bytes())
	}
	if flusher, ok := trw.ginWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (pm *ProxyManager) ollamaChatHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		var ollamaReq OllamaChatRequest
		if err := c.ShouldBindJSON(&ollamaReq); err != nil {
			pm.sendOllamaError(c, http.StatusBadRequest, fmt.Sprintf("Invalid request: %v", err))
			return
		}

		// Validate tool request
		if err := validateToolRequest(&ollamaReq); err != nil {
			pm.sendOllamaError(c, http.StatusBadRequest, err.Error())
			return
		}

		// Normalize keep_alive field to handle both numeric and string inputs
		normalizedKeepAlive := normalizeKeepAlive(ollamaReq.KeepAlive)
		if normalizedKeepAlive != "" {
			ollamaReq.KeepAlive = normalizedKeepAlive
		}

		if ollamaReq.Model == "" {
			pm.sendOllamaError(c, http.StatusBadRequest, "Model name is required.")
			return
		}

		pg, realModelName, err := pm.swapProcessGroup(ollamaReq.Model)
		if err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error selecting model process: %v", err))
			return
		}

		process, ok := pg.processes[realModelName]
		if !ok {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Process for model %s not found in group %s", realModelName, pg.id))
			return
		}

		openAIMessages := ollamaMessagesToOpenAI(ollamaReq.Messages)
		openAITools := ollamaToolsToOpenAI(ollamaReq.Tools)
		modelNameToUse := realModelName
		if pm.config.Models[realModelName].UseModelName != "" {
			modelNameToUse = pm.config.Models[realModelName].UseModelName
		}

		isStreaming := ollamaReq.Stream != nil && *ollamaReq.Stream
		opts := &createOpenAIRequestBodyOptions{
			Think:  ollamaReq.Think,
			Format: ollamaReq.Format,
		}
		openAIReqBodyBytes, err := createOpenAIRequestBody(modelNameToUse, openAIMessages, isStreaming, ollamaReq.Options, openAITools, ollamaReq.ToolChoice, opts)
		if err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error creating OpenAI request: %v", err))
			return
		}

		proxyDestReq, err := http.NewRequestWithContext(c.Request.Context(), "POST", "/v1/chat/completions", bytes.NewBuffer(openAIReqBodyBytes))
		if err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error creating internal request: %v", err))
			return
		}
		proxyDestReq.Header.Set("Content-Type", "application/json")
		proxyDestReq.Header.Set("Accept", "application/json, text/event-stream")
		proxyDestReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(openAIReqBodyBytes)))

		if isStreaming {
			c.Header("Content-Type", "application/x-ndjson")
			c.Header("Transfer-Encoding", "chunked")
			c.Header("Cache-Control", "no-cache")
			c.Header("Connection", "keep-alive")

			trw := newTransformingResponseWriter(c.Writer, ollamaReq.Model, true)
			process.ProxyRequest(trw, proxyDestReq)
			trw.Flush()
		} else {
			recorder := httptest.NewRecorder()
			process.ProxyRequest(recorder, proxyDestReq)

			if recorder.Code != http.StatusOK {
				var openAIError struct {
					Error struct {
						Message string `json:"message"`
						Type    string `json:"type"`
					} `json:"error"`
				}
				if json.Unmarshal(recorder.Body.Bytes(), &openAIError) == nil && openAIError.Error.Message != "" {
					pm.sendOllamaError(c, recorder.Code, openAIError.Error.Message)
				} else {
					pm.sendOllamaError(c, recorder.Code, fmt.Sprintf("Upstream error: %s", recorder.Body.String()))
				}
				return
			}

			var openAIResp OpenAIChatCompletionResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &openAIResp); err != nil {
				pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error parsing OpenAI response: %v. Body: %s", err, recorder.Body.String()))
				return
			}

			if len(openAIResp.Choices) == 0 {
				pm.sendOllamaError(c, http.StatusInternalServerError, "OpenAI response contained no choices.")
				return
			}

			choice := openAIResp.Choices[0]
			message := OllamaMessage{
				Role:     openAIRoleToOllama(choice.Message.Role),
				Content:  choice.Message.Content,
				Thinking: choice.Message.ReasoningContent,
			}

			// Handle tool calls in the response
			if len(choice.Message.ToolCalls) > 0 {
				message.ToolCalls = openAIToolCallsToOllama(choice.Message.ToolCalls)
			}

			ollamaFinalResp := OllamaChatResponse{
				Model:           ollamaReq.Model,
				CreatedAt:       time.Unix(openAIResp.Created, 0).UTC(),
				Message:         message,
				Done:            true,
				DoneReason:      openAIFinishReasonToOllama(choice.FinishReason),
				TotalDuration:   0,
				LoadDuration:    0,
				PromptEvalCount: openAIResp.Usage.PromptTokens,
				EvalCount:       openAIResp.Usage.CompletionTokens,
			}

			// CORS handling (avoid duplicate header)
			if origin := c.Request.Header.Get("Origin"); origin != "" {
				if _, exists := c.Writer.Header()["Access-Control-Allow-Origin"]; !exists {
					c.Header("Access-Control-Allow-Origin", origin)
				}
			}

			c.JSON(http.StatusOK, ollamaFinalResp)
		}
	}
}

func (pm *ProxyManager) ollamaGenerateHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		var ollamaReq OllamaGenerateRequest
		if err := c.ShouldBindJSON(&ollamaReq); err != nil {
			pm.sendOllamaError(c, http.StatusBadRequest, fmt.Sprintf("Invalid request: %v", err))
			return
		}

		// Normalize keep_alive field to handle both numeric and string inputs
		normalizedKeepAlive := normalizeKeepAlive(ollamaReq.KeepAlive)
		if normalizedKeepAlive != "" {
			ollamaReq.KeepAlive = normalizedKeepAlive
		}

		if ollamaReq.Model == "" {
			pm.sendOllamaError(c, http.StatusBadRequest, "Model name is required.")
			return
		}
		if ollamaReq.Raw {
			pm.sendOllamaError(c, http.StatusNotImplemented, "Raw mode for /api/generate is not implemented.")
			return
		}
		if len(ollamaReq.Images) > 0 {
			pm.sendOllamaError(c, http.StatusNotImplemented, "Image input for /api/generate is not implemented.")
			return
		}

		pg, realModelName, err := pm.swapProcessGroup(ollamaReq.Model)
		if err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error selecting model process: %v", err))
			return
		}

		process, ok := pg.processes[realModelName]
		if !ok {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Process for model %s not found in group %s", realModelName, pg.id))
			return
		}

		modelNameToUse := realModelName
		if pm.config.Models[realModelName].UseModelName != "" {
			modelNameToUse = pm.config.Models[realModelName].UseModelName
		}

		isStreaming := ollamaReq.Stream != nil && *ollamaReq.Stream
		fullPrompt := ollamaReq.Prompt
		if ollamaReq.System != "" {
			fullPrompt = ollamaReq.System + "\n\n" + ollamaReq.Prompt
		}

		openAIReqBodyBytes, err := createOpenAILegacyCompletionRequestBody(modelNameToUse, fullPrompt, isStreaming, ollamaReq.Options)
		if err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error creating OpenAI request: %v", err))
			return
		}

		proxyDestReq, err := http.NewRequestWithContext(c.Request.Context(), "POST", "/v1/completions", bytes.NewBuffer(openAIReqBodyBytes))
		if err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error creating internal request: %v", err))
			return
		}
		proxyDestReq.Header.Set("Content-Type", "application/json")
		proxyDestReq.Header.Set("Accept", "application/json, text/event-stream")
		proxyDestReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(openAIReqBodyBytes)))

		if isStreaming {
			c.Header("Content-Type", "application/x-ndjson")
			c.Header("Transfer-Encoding", "chunked")
			c.Header("Cache-Control", "no-cache")
			c.Header("Connection", "keep-alive")

			trw := newTransformingResponseWriter(c.Writer, ollamaReq.Model, false)
			process.ProxyRequest(trw, proxyDestReq)
			trw.Flush()
		} else {
			recorder := httptest.NewRecorder()
			process.ProxyRequest(recorder, proxyDestReq)

			if recorder.Code != http.StatusOK {
				var openAIError struct {
					Error struct {
						Message string `json:"message"`
					} `json:"error"`
				}
				if json.Unmarshal(recorder.Body.Bytes(), &openAIError) == nil && openAIError.Error.Message != "" {
					pm.sendOllamaError(c, recorder.Code, openAIError.Error.Message)
				} else {
					pm.sendOllamaError(c, recorder.Code, fmt.Sprintf("Upstream error: %s", recorder.Body.String()))
				}
				return
			}

			var openAIResp OpenAICompletionResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &openAIResp); err != nil {
				pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error parsing OpenAI response: %v. Body: %s", err, recorder.Body.String()))
				return
			}

			if len(openAIResp.Choices) == 0 {
				pm.sendOllamaError(c, http.StatusInternalServerError, "OpenAI response contained no choices.")
				return
			}

			choice := openAIResp.Choices[0]
			ollamaFinalResp := OllamaGenerateResponse{
				Model:           ollamaReq.Model,
				CreatedAt:       time.Unix(openAIResp.Created, 0).UTC(),
				Response:        choice.Text,
				Done:            true,
				DoneReason:      openAIFinishReasonToOllama(choice.FinishReason),
				PromptEvalCount: openAIResp.Usage.PromptTokens,
				EvalCount:       openAIResp.Usage.CompletionTokens,
			}

			// CORS handling (avoid duplicate header)
			if origin := c.Request.Header.Get("Origin"); origin != "" {
				if _, exists := c.Writer.Header()["Access-Control-Allow-Origin"]; !exists {
					c.Header("Access-Control-Allow-Origin", origin)
				}
			}

			c.JSON(http.StatusOK, ollamaFinalResp)
		}
	}
}

func (pm *ProxyManager) ollamaEmbedHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req OllamaEmbedRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			pm.sendOllamaError(c, http.StatusBadRequest, fmt.Sprintf("Invalid request: %v", err))
			return
		}

		// Normalize keep_alive field to handle both numeric and string inputs
		normalizedKeepAlive := normalizeKeepAlive(req.KeepAlive)
		if normalizedKeepAlive != "" {
			req.KeepAlive = normalizedKeepAlive
		}

		if req.Model == "" {
			pm.sendOllamaError(c, http.StatusBadRequest, "Model name is required.")
			return
		}

		pg, realModelName, err := pm.swapProcessGroup(req.Model)
		if err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error selecting model process: %v", err))
			return
		}
		process, ok := pg.processes[realModelName]
		if !ok {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Process for model %s not found in group %s", realModelName, pg.id))
			return
		}

		modelNameToUse := realModelName
		if pm.config.Models[realModelName].UseModelName != "" {
			modelNameToUse = pm.config.Models[realModelName].UseModelName
		}

		// Prepare OpenAI embeddings request
		openAIReq := map[string]interface{}{
			"model": modelNameToUse,
		}
		switch v := req.Input.(type) {
		case string:
			openAIReq["input"] = v
		case []interface{}:
			openAIReq["input"] = v
		default:
			openAIReq["input"] = req.Input
		}
		if req.Options != nil {
			for k, v := range req.Options {
				openAIReq[k] = v
			}
		}

		openAIReqBody, err := json.Marshal(openAIReq)
		if err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error marshaling OpenAI request: %v", err))
			return
		}

		proxyDestReq, err := http.NewRequestWithContext(c.Request.Context(), "POST", "/v1/embeddings", bytes.NewBuffer(openAIReqBody))
		if err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error creating internal request: %v", err))
			return
		}

		proxyDestReq.Header.Set("Content-Type", "application/json")
		proxyDestReq.Header.Set("Accept", "application/json")
		proxyDestReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(openAIReqBody)))

		recorder := httptest.NewRecorder()
		process.ProxyRequest(recorder, proxyDestReq)

		// CORS handling
		if origin := c.Request.Header.Get("Origin"); origin != "" {
			if _, exists := c.Writer.Header()["Access-Control-Allow-Origin"]; !exists {
				c.Header("Access-Control-Allow-Origin", origin)
			}
		}

		if recorder.Code != http.StatusOK {
			var openAIError struct {
				Error struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			if json.Unmarshal(recorder.Body.Bytes(), &openAIError) == nil && openAIError.Error.Message != "" {
				pm.sendOllamaError(c, recorder.Code, openAIError.Error.Message)
			} else {
				pm.sendOllamaError(c, recorder.Code, fmt.Sprintf("Upstream error: %s", recorder.Body.String()))
			}
			return
		}

		// Parse OpenAI response and transform to Ollama format
		var openAIResp struct {
			Object string `json:"object"`
			Model  string `json:"model"`
			Data   []struct {
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
			Usage struct {
				PromptTokens int `json:"prompt_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(recorder.Body.Bytes(), &openAIResp); err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error parsing OpenAI response: %v. Body: %s", err, recorder.Body.String()))
			return
		}

		embeddings := make([][]float32, len(openAIResp.Data))
		for i, d := range openAIResp.Data {
			embeddings[i] = d.Embedding
		}

		resp := OllamaEmbedResponse{
			Model:           req.Model,
			Embeddings:      embeddings,
			PromptEvalCount: openAIResp.Usage.PromptTokens,
		}

		c.JSON(http.StatusOK, resp)
	}
}

func (pm *ProxyManager) ollamaLegacyEmbeddingsHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req OllamaLegacyEmbeddingsRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			pm.sendOllamaError(c, http.StatusBadRequest, fmt.Sprintf("Invalid request: %v", err))
			return
		}

		// Normalize keep_alive field to handle both numeric and string inputs
		normalizedKeepAlive := normalizeKeepAlive(req.KeepAlive)
		if normalizedKeepAlive != "" {
			req.KeepAlive = normalizedKeepAlive
		}

		if req.Model == "" {
			pm.sendOllamaError(c, http.StatusBadRequest, "Model name is required.")
			return
		}
		if req.Prompt == "" {
			pm.sendOllamaError(c, http.StatusBadRequest, "Prompt is required.")
			return
		}

		pg, realModelName, err := pm.swapProcessGroup(req.Model)
		if err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error selecting model process: %v", err))
			return
		}
		process, ok := pg.processes[realModelName]
		if !ok {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Process for model %s not found in group %s", realModelName, pg.id))
			return
		}

		modelNameToUse := realModelName
		if pm.config.Models[realModelName].UseModelName != "" {
			modelNameToUse = pm.config.Models[realModelName].UseModelName
		}

		// Prepare OpenAI embeddings request
		openAIReq := map[string]interface{}{
			"model": modelNameToUse,
			"input": req.Prompt,
		}
		if req.Options != nil {
			for k, v := range req.Options {
				openAIReq[k] = v
			}
		}

		openAIReqBody, err := json.Marshal(openAIReq)
		if err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error marshaling OpenAI request: %v", err))
			return
		}

		proxyDestReq, err := http.NewRequestWithContext(c.Request.Context(), "POST", "/v1/embeddings", bytes.NewBuffer(openAIReqBody))
		if err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error creating internal request: %v", err))
			return
		}
		proxyDestReq.Header.Set("Content-Type", "application/json")
		proxyDestReq.Header.Set("Accept", "application/json")
		proxyDestReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(openAIReqBody)))

		recorder := httptest.NewRecorder()
		process.ProxyRequest(recorder, proxyDestReq)

		// CORS handling
		if origin := c.Request.Header.Get("Origin"); origin != "" {
			if _, exists := c.Writer.Header()["Access-Control-Allow-Origin"]; !exists {
				c.Header("Access-Control-Allow-Origin", origin)
			}
		}

		if recorder.Code != http.StatusOK {
			var openAIError struct {
				Error struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			if json.Unmarshal(recorder.Body.Bytes(), &openAIError) == nil && openAIError.Error.Message != "" {
				pm.sendOllamaError(c, recorder.Code, openAIError.Error.Message)
			} else {
				pm.sendOllamaError(c, recorder.Code, fmt.Sprintf("Upstream error: %s", recorder.Body.String()))
			}
			return
		}

		// Parse OpenAI response and transform to Ollama legacy format
		var openAIResp struct {
			Data []struct {
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
		}
		if err := json.Unmarshal(recorder.Body.Bytes(), &openAIResp); err != nil {
			pm.sendOllamaError(c, http.StatusInternalServerError, fmt.Sprintf("Error parsing OpenAI response: %v. Body: %s", err, recorder.Body.String()))
			return
		}
		if len(openAIResp.Data) == 0 {
			pm.sendOllamaError(c, http.StatusInternalServerError, "OpenAI response contained no embeddings.")
			return
		}

		resp := OllamaLegacyEmbeddingsResponse{
			Embedding: openAIResp.Data[0].Embedding,
		}

		c.JSON(http.StatusOK, resp)
	}
}

// OllamaErrorResponse is the standard error format for Ollama API.
type OllamaErrorResponse struct {
	Error string `json:"error"`
}

// OllamaVersionResponse defines the structure for the /api/version endpoint.
type OllamaVersionResponse struct {
	Version string `json:"version"`
}

// OllamaGenerateRequest describes a request to /api/generate.
type OllamaGenerateRequest struct {
	Model     string                 `json:"model"`
	Prompt    string                 `json:"prompt"`
	System    string                 `json:"system,omitempty"`
	Template  string                 `json:"template,omitempty"`
	Context   []int                  `json:"context,omitempty"`
	Stream    *bool                  `json:"stream,omitempty"`
	Raw       bool                   `json:"raw,omitempty"`
	Format    string                 `json:"format,omitempty"`
	Images    []string               `json:"images,omitempty"`
	KeepAlive interface{}            `json:"keep_alive,omitempty"`
	Options   map[string]interface{} `json:"options,omitempty"`
}

// OllamaGenerateResponse is the response from /api/generate.
type OllamaGenerateResponse struct {
	Model              string    `json:"model"`
	CreatedAt          time.Time `json:"created_at"`
	Response           string    `json:"response,omitempty"`
	Done               bool      `json:"done"`
	DoneReason         string    `json:"done_reason,omitempty"`
	Context            []int     `json:"context,omitempty"`
	TotalDuration      int64     `json:"total_duration,omitempty"`
	LoadDuration       int64     `json:"load_duration,omitempty"`
	PromptEvalCount    int       `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64     `json:"prompt_eval_duration,omitempty"`
	EvalCount          int       `json:"eval_count,omitempty"`
	EvalDuration       int64     `json:"eval_duration,omitempty"`
}

// Tool definition types
type OllamaTool struct {
	Type     string             `json:"type"`
	Function OllamaToolFunction `json:"function"`
}

type OllamaToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// Tool call types - Compatible with both Ollama native and OpenAI formats
type OllamaToolCall struct {
	ID       string             `json:"id,omitempty"`   // Optional for compatibility with older Ollama
	Type     string             `json:"type,omitempty"` // Optional - Zed doesn't send this
	Function OllamaToolCallFunc `json:"function"`
}

type OllamaToolCallFunc struct {
	Index     int                    `json:"index,omitempty"` // Optional
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// OllamaMessage represents a single message in a chat.
type OllamaMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	Thinking   string           `json:"thinking,omitempty"` // Reasoning trace for thinking models (from OpenAI reasoning_content)
	Images     []string         `json:"images,omitempty"`
	ToolCalls  []OllamaToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"` // For tool role messages (OpenAI style)
	ToolName   string           `json:"tool_name,omitempty"`    // For tool role messages (Ollama native style)
}

// OllamaChatRequest describes a request to /api/chat.
type OllamaChatRequest struct {
	Model      string                 `json:"model"`
	Messages   []OllamaMessage        `json:"messages"`
	Stream     *bool                  `json:"stream,omitempty"`
	Format     interface{}            `json:"format,omitempty"` // string "json" or JSON Schema object
	KeepAlive  interface{}            `json:"keep_alive,omitempty"`
	Options    map[string]interface{} `json:"options,omitempty"`
	Tools      []OllamaTool           `json:"tools,omitempty"`
	ToolChoice interface{}            `json:"tool_choice,omitempty"`
	Think      *bool                  `json:"think,omitempty"` // Enable/disable thinking mode for reasoning models
}

// OllamaChatResponse is the response from /api/chat.
type OllamaChatResponse struct {
	Model              string        `json:"model"`
	CreatedAt          time.Time     `json:"created_at"`
	Message            OllamaMessage `json:"message,omitempty"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason,omitempty"`
	TotalDuration      int64         `json:"total_duration,omitempty"`
	LoadDuration       int64         `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64         `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       int64         `json:"eval_duration,omitempty"`
}

// OllamaListTagsResponse is the response from /api/tags.
type OllamaListTagsResponse struct {
	Models []OllamaModelResponse `json:"models"`
}

// OllamaModelResponse describes a single model in the list.
type OllamaModelResponse struct {
	Name       string             `json:"name"`
	Model      string             `json:"model"`
	ModifiedAt time.Time          `json:"modified_at"`
	Size       int64              `json:"size"`
	Digest     string             `json:"digest"`
	Details    OllamaModelDetails `json:"details"`
}

// OllamaModelDetails provides more details about a model.
type OllamaModelDetails struct {
	ParentModel       string   `json:"parent_model,omitempty"`
	Format            string   `json:"format,omitempty"`
	Family            string   `json:"family,omitempty"`
	Families          []string `json:"families,omitempty"`
	ParameterSize     string   `json:"parameter_size,omitempty"`
	QuantizationLevel string   `json:"quantization_level,omitempty"`
}

type OllamaTensor struct {
	Name  string   `json:"name"`
	Type  string   `json:"type"`
	Shape []uint64 `json:"shape"`
}

// OllamaShowRequest is the request for /api/show.
type OllamaShowRequest struct {
	Model string `json:"model,omitempty"`
	Name  string `json:"name,omitempty"`
}

// OllamaShowResponse is the response from /api/show.
type OllamaShowResponse struct {
	License       string             `json:"license,omitempty"`
	Modelfile     string             `json:"modelfile,omitempty"`
	Parameters    string             `json:"parameters,omitempty"`
	Template      string             `json:"template,omitempty"`
	System        string             `json:"system,omitempty"`
	Details       OllamaModelDetails `json:"details,omitempty"`
	Messages      []OllamaMessage    `json:"messages,omitempty"`
	ModelInfo     map[string]any     `json:"model_info,omitempty"`
	ProjectorInfo map[string]any     `json:"projector_info,omitempty"`
	Tensors       []OllamaTensor     `json:"tensors,omitempty"`
	Capabilities  []string           `json:"capabilities,omitempty"`
	ModifiedAt    time.Time          `json:"modified_at,omitempty"`
}

// OllamaProcessResponse is the response from /api/ps.
type OllamaProcessResponse struct {
	Models []OllamaProcessModelResponse `json:"models"`
}

// OllamaProcessModelResponse describes a running model process.
type OllamaProcessModelResponse struct {
	Name      string             `json:"name"`
	Model     string             `json:"model"`
	Size      int64              `json:"size"`
	Digest    string             `json:"digest"`
	Details   OllamaModelDetails `json:"details"`
	ExpiresAt time.Time          `json:"expires_at"`
	SizeVRAM  int64              `json:"size_vram"`
}

// OllamaEmbedRequest describes a request to /api/embed.
type OllamaEmbedRequest struct {
	Model     string                 `json:"model"`
	Input     interface{}            `json:"input"` // string or []string
	Truncate  *bool                  `json:"truncate,omitempty"`
	Options   map[string]interface{} `json:"options,omitempty"`
	KeepAlive interface{}            `json:"keep_alive,omitempty"`
}

// OllamaEmbedResponse is the response from /api/embed.
type OllamaEmbedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float32 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration,omitempty"`
	LoadDuration    int64       `json:"load_duration,omitempty"`
	PromptEvalCount int         `json:"prompt_eval_count,omitempty"`
}

// OllamaLegacyEmbeddingsRequest describes a request to /api/embeddings.
type OllamaLegacyEmbeddingsRequest struct {
	Model     string                 `json:"model"`
	Prompt    string                 `json:"prompt"`
	Options   map[string]interface{} `json:"options,omitempty"`
	KeepAlive interface{}            `json:"keep_alive,omitempty"`
}

// OllamaLegacyEmbeddingsResponse is the response from /api/embeddings.
type OllamaLegacyEmbeddingsResponse struct {
	Embedding []float32 `json:"embedding"`
}

// --- Helper types for transforming OpenAI stream to Ollama stream ---

// OpenAIChatCompletionStreamChoiceDelta is part of an OpenAI stream event.
type OpenAIChatCompletionStreamChoiceDelta struct {
	Content          string                      `json:"content,omitempty"`
	ReasoningContent string                      `json:"reasoning_content,omitempty"` // For thinking/reasoning models
	Role             string                      `json:"role,omitempty"`
	ToolCalls        []OpenAIStreamToolCallDelta `json:"tool_calls,omitempty"`
}

// OpenAIStreamToolCallDelta represents a tool call delta in a streaming response
type OpenAIStreamToolCallDelta struct {
	Index    int                          `json:"index"`
	ID       string                       `json:"id,omitempty"`
	Type     string                       `json:"type,omitempty"`
	Function OpenAIStreamToolCallFunction `json:"function,omitempty"`
}

// OpenAIStreamToolCallFunction represents the function part of a streaming tool call
type OpenAIStreamToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// OpenAIChatCompletionStreamChoice is part of an OpenAI stream event.
type OpenAIChatCompletionStreamChoice struct {
	Index        int                                   `json:"index"`
	Delta        OpenAIChatCompletionStreamChoiceDelta `json:"delta"`
	FinishReason string                                `json:"finish_reason,omitempty"`
}

// OpenAIStreamingChatResponse is a typical OpenAI chat completion stream event.
type OpenAIStreamingChatResponse struct {
	ID      string                             `json:"id"`
	Object  string                             `json:"object"`
	Created int64                              `json:"created"`
	Model   string                             `json:"model"`
	Choices []OpenAIChatCompletionStreamChoice `json:"choices"`
	Usage   *OpenAIUsage                       `json:"usage,omitempty"`
}

// OpenAICompletionStreamChoice is part of an OpenAI legacy completion stream event.
type OpenAICompletionStreamChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// OpenAIStreamingCompletionResponse is a typical OpenAI legacy completion stream event.
type OpenAIStreamingCompletionResponse struct {
	ID      string                         `json:"id"`
	Object  string                         `json:"object"`
	Created int64                          `json:"created"`
	Model   string                         `json:"model"`
	Choices []OpenAICompletionStreamChoice `json:"choices"`
	Usage   *OpenAIUsage                   `json:"usage,omitempty"`
}

// OpenAIUsage represents token usage statistics.
type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// OpenAIChatCompletionResponse is a non-streaming OpenAI chat response.
type OpenAIChatCompletionResponse struct {
	ID      string                               `json:"id"`
	Object  string                               `json:"object"`
	Created int64                                `json:"created"`
	Model   string                               `json:"model"`
	Choices []OpenAIChatCompletionResponseChoice `json:"choices"`
	Usage   OpenAIUsage                          `json:"usage"`
}

// OpenAIChatCompletionMessage is the message structure in a non-streaming OpenAI response.
type OpenAIChatCompletionMessage struct {
	Role             string           `json:"role"`
	Content          string           `json:"content"`
	ReasoningContent string           `json:"reasoning_content,omitempty"` // For thinking/reasoning models
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
}

// OpenAIToolCall represents a tool call in OpenAI format
type OpenAIToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function OpenAIToolCallFunction `json:"function"`
}

// OpenAIToolCallFunction represents the function part of a tool call
type OpenAIToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// OpenAIChatCompletionResponseChoice is part of a non-streaming OpenAI chat response.
type OpenAIChatCompletionResponseChoice struct {
	Index        int                         `json:"index"`
	Message      OpenAIChatCompletionMessage `json:"message"`
	FinishReason string                      `json:"finish_reason"`
}

// OpenAICompletionResponse is a non-streaming OpenAI legacy completion response.
type OpenAICompletionResponse struct {
	ID      string                         `json:"id"`
	Object  string                         `json:"object"`
	Created int64                          `json:"created"`
	Model   string                         `json:"model"`
	Choices []OpenAICompletionStreamChoice `json:"choices"`
	Usage   OpenAIUsage                    `json:"usage"`
}

func openAIFinishReasonToOllama(reason string) string {
	switch reason {
	case "stop":
		return "stop"
	case "length":
		return "length"
	case "content_filter":
		return "content_filter"
	case "tool_calls":
		return "tool_calls"
	default:
		if reason != "" {
			return "unknown"
		}
		return ""
	}
}

func openAIRoleToOllama(role string) string {
	switch role {
	case "system":
		return "system"
	case "user":
		return "user"
	case "assistant":
		return "assistant"
	default:
		return role
	}
}

func ollamaMessagesToOpenAI(ollamaMsgs []OllamaMessage) []map[string]interface{} {
	openAIMsgs := make([]map[string]interface{}, len(ollamaMsgs))
	for i, msg := range ollamaMsgs {
		openAIMsg := map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		}

		// Handle tool calls from assistant
		// Filter out invalid tool calls (empty function names) which can occur
		// when models hallucinate extra tool calls
		if len(msg.ToolCalls) > 0 {
			openAIToolCalls := make([]map[string]interface{}, 0, len(msg.ToolCalls))
			validIndex := 0
			for _, tc := range msg.ToolCalls {
				// Skip invalid tool calls with empty function names
				if tc.Function.Name == "" {
					continue
				}

				argsJSON, _ := json.Marshal(tc.Function.Arguments)

				// Generate ID if missing (for compatibility with clients like Zed)
				toolID := tc.ID
				if toolID == "" {
					toolID = fmt.Sprintf("call_%d_%d", i, validIndex)
				}

				// Default type to "function" if not provided
				toolType := tc.Type
				if toolType == "" {
					toolType = "function"
				}

				openAIToolCalls = append(openAIToolCalls, map[string]interface{}{
					"id":   toolID,
					"type": toolType,
					"function": map[string]interface{}{
						"name":      tc.Function.Name,
						"arguments": string(argsJSON),
					},
				})
				validIndex++
			}
			if len(openAIToolCalls) > 0 {
				openAIMsg["tool_calls"] = openAIToolCalls
			}
		}

		// Handle tool role messages
		// Support both OpenAI style (tool_call_id) and Ollama native style (tool_name)
		if msg.Role == "tool" {
			if msg.ToolCallID != "" {
				openAIMsg["tool_call_id"] = msg.ToolCallID
			} else if msg.ToolName != "" {
				// Ollama native format uses tool_name instead of tool_call_id
				// Generate a synthetic ID based on the tool name for OpenAI compatibility
				openAIMsg["tool_call_id"] = fmt.Sprintf("call_%s_%d", msg.ToolName, i)
			}
			if msg.Content == "" {
				openAIMsg["content"] = "null"
			}
		}

		openAIMsgs[i] = openAIMsg
	}
	return openAIMsgs
}

func ollamaToolsToOpenAI(ollamaTools []OllamaTool) []map[string]interface{} {
	if len(ollamaTools) == 0 {
		return nil
	}

	openAITools := make([]map[string]interface{}, len(ollamaTools))
	for i, tool := range ollamaTools {
		openAITools[i] = map[string]interface{}{
			"type": tool.Type,
			"function": map[string]interface{}{
				"name":        tool.Function.Name,
				"description": tool.Function.Description,
				"parameters":  tool.Function.Parameters, // Pass as object, not JSON string
			},
		}
	}
	return openAITools
}

// createOpenAIRequestBodyOptions holds optional parameters for createOpenAIRequestBody
type createOpenAIRequestBodyOptions struct {
	Think  *bool       // Ollama think parameter -> chat_template_kwargs.enable_thinking
	Format interface{} // Ollama format parameter (string "json" or JSON Schema object)
}

func createOpenAIRequestBody(modelName string, messages []map[string]interface{}, stream bool, options map[string]interface{}, tools []map[string]interface{}, toolChoice interface{}, opts *createOpenAIRequestBodyOptions) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model":    modelName,
		"messages": messages,
		"stream":   stream,
	}

	if tools != nil {
		requestBody["tools"] = tools
	}

	if toolChoice != nil {
		requestBody["tool_choice"] = toolChoice
	}

	if options != nil {
		for k, v := range options {
			if _, exists := requestBody[k]; !exists {
				requestBody[k] = v
			}
		}
	}

	// Handle Ollama-specific options
	if opts != nil {
		// Translate Ollama's think parameter to llama-server's chat_template_kwargs
		if opts.Think != nil {
			requestBody["chat_template_kwargs"] = map[string]interface{}{
				"enable_thinking": *opts.Think,
			}
		}

		// Handle format parameter
		if opts.Format != nil {
			switch f := opts.Format.(type) {
			case string:
				// Simple "json" format
				if f == "json" {
					requestBody["response_format"] = map[string]interface{}{
						"type": "json_object",
					}
				}
			case map[string]interface{}:
				// JSON Schema object for structured outputs
				requestBody["response_format"] = map[string]interface{}{
					"type":   "json_schema",
					"schema": f,
				}
			}
		}
	}

	return json.Marshal(requestBody)
}

func openAIToolCallsToOllama(toolCalls []OpenAIToolCall) []OllamaToolCall {
	if len(toolCalls) == 0 {
		return nil
	}

	ollamaToolCalls := make([]OllamaToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		var args map[string]interface{}
		json.Unmarshal([]byte(tc.Function.Arguments), &args)

		ollamaToolCalls[i] = OllamaToolCall{
			ID:   tc.ID, // CRITICAL: Preserve OpenAI ID
			Type: tc.Type,
			Function: OllamaToolCallFunc{
				Index:     i,
				Name:      tc.Function.Name,
				Arguments: args,
			},
		}
	}

	return ollamaToolCalls
}

func validateToolRequest(req *OllamaChatRequest) error {
	// Validate tool definitions
	for i, tool := range req.Tools {
		if tool.Type != "function" {
			return fmt.Errorf("tool %d: only 'function' type is supported", i)
		}
		if tool.Function.Name == "" {
			return fmt.Errorf("tool %d: missing function name", i)
		}
	}

	// Validate messages - be lenient for compatibility with various clients
	// and to handle models that hallucinate invalid tool calls
	//
	// First pass: filter out invalid tool calls with empty function names
	for i, msg := range req.Messages {
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			validToolCalls := make([]OllamaToolCall, 0, len(msg.ToolCalls))
			for _, tc := range msg.ToolCalls {
				if tc.Function.Name != "" {
					validToolCalls = append(validToolCalls, tc)
				}
				// Silently skip invalid tool calls with empty function names
			}
			// Update the message with only valid tool calls
			req.Messages[i].ToolCalls = validToolCalls
		}
	}

	// Second pass: filter out tool response messages that have empty tool_name
	// These correspond to the hallucinated tool calls we filtered above
	validMessages := make([]OllamaMessage, 0, len(req.Messages))
	for _, msg := range req.Messages {
		// Skip tool responses with empty identifiers (responses to hallucinated tool calls)
		if msg.Role == "tool" && msg.ToolCallID == "" && msg.ToolName == "" {
			continue // Skip this message entirely
		}
		validMessages = append(validMessages, msg)
	}
	req.Messages = validMessages

	return nil
}

func createOpenAILegacyCompletionRequestBody(modelName string, prompt string, stream bool, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model":  modelName,
		"prompt": prompt,
		"stream": stream,
	}
	if options != nil {
		for k, v := range options {
			if _, exists := requestBody[k]; !exists {
				requestBody[k] = v
			}
		}
	}
	return json.Marshal(requestBody)
}
