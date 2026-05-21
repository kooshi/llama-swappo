package proxy

import (
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/mostlygeek/llama-swap/proxy/config"
)

// ServerArgs holds information parsed or inferred from server command line arguments.
type ServerArgs struct {
	Architecture      string
	ContextLength     int
	PredictLength     int // max output tokens (-n/--predict)
	Capabilities      []string
	Family            string
	ParameterSize     string
	QuantizationLevel string
	CmdAlias          string
	CmdModelPath      string // Base name of the model file from --model arg
}

// ServerArgParser defines an interface for parsing server command line arguments.
type ServerArgParser interface {
	Parse(cmdStr string, modelID string) ServerArgs
}

// LlamaServerParser implements ServerArgParser for llama-server.
type LlamaServerParser struct{}

var (
	architecturePatterns = map[string]*regexp.Regexp{
		"command-r": regexp.MustCompile(`(?i)command-r`),
		"gemma2":    regexp.MustCompile(`(?i)gemma2`),
		"gemma3":    regexp.MustCompile(`(?i)gemma3`),
		"gemma":     regexp.MustCompile(`(?i)gemma`),
		"llama4":    regexp.MustCompile(`(?i)llama-?4`),
		"llama3":    regexp.MustCompile(`(?i)llama-?3`),
		"llama":     regexp.MustCompile(`(?i)llama`),
		"mistral3":  regexp.MustCompile(`(?i)mistral-?3`),
		"mistral":   regexp.MustCompile(`(?i)mistral`),
		"phi3":      regexp.MustCompile(`(?i)phi-?3`),
		"phi":       regexp.MustCompile(`(?i)phi`),
		"qwen2.5vl": regexp.MustCompile(`(?i)qwen-?2\.5-?vl`),
		"qwen3":     regexp.MustCompile(`(?i)qwen-?3`),
		"qwen2":     regexp.MustCompile(`(?i)qwen-?2`),
		"qwen":      regexp.MustCompile(`(?i)qwen`),
		"bert":      regexp.MustCompile(`(?i)bert`),
		"clip":      regexp.MustCompile(`(?i)clip`),
	}
	orderedArchKeys = []string{
		"command-r", "gemma3", "gemma2", "gemma", "llama4", "llama3", "llama",
		"mistral3", "mistral", "phi3", "phi", "qwen2.5vl", "qwen3", "qwen2", "qwen",
		"bert", "clip",
	}

	parameterSizePattern = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?(?:x\d+)?)[BMGT]?B`)
	quantizationPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)IQ[1-4]_(XXS|XS|S|M|NL)`),
		regexp.MustCompile(`(?i)Q[2-8]_(0|1|[KSLM]+(?:_[KSLM]+)?)`),
		regexp.MustCompile(`(?i)BPW\d+`),
		regexp.MustCompile(`(?i)GGML_TYPE_Q[2-8]_\d`),
		regexp.MustCompile(`(?i)F(?:P)?(16|32)`),
		regexp.MustCompile(`(?i)BF16`),
	}
)

func inferPattern(name string, patterns map[string]*regexp.Regexp, orderedKeys []string) string {
	nameLower := strings.ToLower(name)
	for _, key := range orderedKeys {
		pattern, ok := patterns[key]
		if !ok || pattern == nil {
			continue
		}
		if pattern.MatchString(nameLower) {
			return key
		}
	}
	return "unknown"
}

func inferQuantizationLevelFromName(name string) string {
	for _, pattern := range quantizationPatterns {
		match := pattern.FindString(name)
		if match != "" {
			return strings.ToUpper(match)
		}
	}
	return "unknown"
}

func inferParameterSizeFromName(name string) string {
	match := parameterSizePattern.FindStringSubmatch(name)
	if len(match) > 0 {
		return strings.ToUpper(match[0])
	}
	return "unknown"
}

func inferFamilyFromName(nameForInference string, currentArch string) string {
	if currentArch != "unknown" && currentArch != "" {
		re := regexp.MustCompile(`^([a-zA-Z_][a-zA-Z0-9_]*)`)
		match := re.FindStringSubmatch(currentArch)
		if len(match) > 1 {
			potentialFamily := strings.ToLower(match[1])
			knownFamilies := []string{"llama", "qwen", "phi", "mistral", "gemma", "command-r", "bert", "clip"}
			for _, kf := range knownFamilies {
				if potentialFamily == kf {
					return kf
				}
			}
			for _, kf := range knownFamilies {
				if strings.ToLower(currentArch) == kf {
					return kf
				}
			}
		}
	}
	orderedFamilyCheckKeys := []string{"command-r", "gemma", "llama", "mistral", "phi", "qwen", "bert", "clip"}
	familyPatterns := make(map[string]*regexp.Regexp)
	for _, key := range orderedFamilyCheckKeys {
		if p, ok := architecturePatterns[key]; ok {
			familyPatterns[key] = p
		}
	}
	return inferPattern(nameForInference, familyPatterns, orderedFamilyCheckKeys)
}

// archChatTemplates maps architecture names to their standard Ollama-format chat templates.
// These correspond to the templates used by the official Ollama project.
var archChatTemplates = map[string]string{
	// Llama3/4: uses special header/eot tokens
	"llama3": "{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n{{ .Content }}<|eot_id|>\n{{- end }}<|start_header_id|>assistant<|end_header_id|>\n\n",
	"llama4": "{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n{{ .Content }}<|eot_id|>\n{{- end }}<|start_header_id|>assistant<|end_header_id|>\n\n",
	// Llama2: legacy chat format
	"llama": "[INST] {{ if .System }}{{ .System }}\n{{ end }}{{ range .Messages }}{{ if eq .Role \"user\" }}{{ .Content }}[/INST]\n{{ else if eq .Role \"assistant\" }}{{ .Content }}</s>[INST] {{ end }}{{ end }}",
	// Gemma3: newer turn format with system support
	"gemma3": "{{- range $i, $_ := .Messages }}\n{{- $last := eq (len (slice $.Messages $i)) 1 }}\n{{- if eq .Role \"user\" }}<start_of_turn>user\n{{- if and (eq $i 1) $.System }}\n{{ $.System }}\n{{ end }}\n{{ .Content }}<end_of_turn>\n{{ else if eq .Role \"assistant\" }}<start_of_turn>model\n{{ .Content }}<end_of_turn>\n{{ end }}\n{{- if $last }}<start_of_turn>model\n{{ end }}\n{{- end }}",
	// Gemma2/Gemma: older turn format
	"gemma2": "{{- $system := \"\" }}\n{{- range .Messages }}\n{{- if eq .Role \"system\" }}{{- if not $system }}{{ $system = .Content }}{{- else }}{{ $system = printf \"%s\\n\\n%s\" $system .Content }}{{- end }}{{- continue }}{{- else if eq .Role \"user\" }}<start_of_turn>user\n{{- if $system }}\n{{ $system }}\n{{- $system = \"\" }}{{- end }}\n{{- else if eq .Role \"assistant\" }}<start_of_turn>model\n{{- end }}\n{{ .Content }}<end_of_turn>\n{{ end }}<start_of_turn>model\n",
	"gemma":  "{{- $system := \"\" }}\n{{- range .Messages }}\n{{- if eq .Role \"system\" }}{{- if not $system }}{{ $system = .Content }}{{- else }}{{ $system = printf \"%s\\n\\n%s\" $system .Content }}{{- end }}{{- continue }}{{- else if eq .Role \"user\" }}<start_of_turn>user\n{{- if $system }}\n{{ $system }}\n{{- $system = \"\" }}{{- end }}\n{{- else if eq .Role \"assistant\" }}<start_of_turn>model\n{{- end }}\n{{ .Content }}<end_of_turn>\n{{ end }}<start_of_turn>model\n",
	// Mistral: uses [INST] tags
	"mistral3": "[INST] {{ range $index, $_ := .Messages }}\n{{- if eq .Role \"system\" }}{{ .Content }}\n\n{{ else if eq .Role \"user\" }}{{ .Content }}[/INST]\n{{- else if eq .Role \"assistant\" }} {{ .Content }}</s>[INST] {{ end }}\n{{- end }}",
	"mistral":  "[INST] {{ range $index, $_ := .Messages }}\n{{- if eq .Role \"system\" }}{{ .Content }}\n\n{{ else if eq .Role \"user\" }}{{ .Content }}[/INST]\n{{- else if eq .Role \"assistant\" }} {{ .Content }}</s>[INST] {{ end }}\n{{- end }}",
	// Phi3/Phi: uses role tokens with <|end|>
	"phi3": "{{- range .Messages }}<|{{ .Role }}|>\n{{ .Content }}<|end|>\n{{ end }}<|assistant|>\n",
	"phi":  "{{- range .Messages }}<|{{ .Role }}|>\n{{ .Content }}<|end|>\n{{ end }}<|assistant|>\n",
	// Command-R: Cohere format
	"command-r": "{{- if or .Tools .System }}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{- if .System }}{{ .System }}{{- end }}<|END_OF_TURN_TOKEN|>{{- end }}\n{{- range .Messages }}\n{{- if eq .Role \"system\" }}{{- continue }}{{- end }}<|START_OF_TURN_TOKEN|>\n{{- if eq .Role \"user\" }}<|USER_TOKEN|>{{ .Content }}\n{{- else if eq .Role \"assistant\" }}<|CHATBOT_TOKEN|>\n{{- if .Content }}{{ .Content }}{{- end }}\n{{- end }}<|END_OF_TURN_TOKEN|>\n{{- end }}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
	// Qwen2/Qwen3/QwenVL: ChatML format
	"qwen2.5vl": "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n",
	"qwen3":     "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n",
	"qwen2":     "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n",
	"qwen":      "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n",
}

// GetArchitectureTemplate returns the Ollama-format chat template for the given architecture.
// Returns an empty string if no template is known for the architecture.
func GetArchitectureTemplate(arch string) string {
	return archChatTemplates[arch]
}

// NewLlamaServerParser creates a new parser for llama-server arguments.
func NewLlamaServerParser() *LlamaServerParser {
	return &LlamaServerParser{}
}

// Parse extracts relevant information from llama-server command string and modelID.
func (p *LlamaServerParser) Parse(cmdStr string, modelID string) ServerArgs {
	parsed := ServerArgs{
		Capabilities: []string{"completion"}, // Default
	}

	args, err := config.SanitizeCommand(cmdStr)
	if err != nil {
		// If sanitization fails, proceed with inference based on modelID only
		parsed.Architecture = inferPattern(modelID, architecturePatterns, orderedArchKeys)
		parsed.Family = inferFamilyFromName(modelID, parsed.Architecture)
		parsed.ParameterSize = inferParameterSizeFromName(modelID)
		parsed.QuantizationLevel = inferQuantizationLevelFromName(modelID)
		return parsed
	}

	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch arg {
		case "-c", "--ctx-size":
			if i+1 < len(args) {
				if valInt, err := strconv.Atoi(args[i+1]); err == nil {
					parsed.ContextLength = valInt
				}
				i++
			}
		case "-n", "--predict":
			if i+1 < len(args) {
				if valInt, err := strconv.Atoi(args[i+1]); err == nil && valInt > 0 {
					parsed.PredictLength = valInt
				}
				i++
			}
		case "-a", "--alias":
			if i+1 < len(args) {
				parsed.CmdAlias = args[i+1]
				i++
			}
		case "-m", "--model":
			if i+1 < len(args) {
				parsed.CmdModelPath = filepath.Base(args[i+1])
				i++
			}
		case "--jinja":
			foundTools := false
			for _, cap := range parsed.Capabilities {
				if cap == "tools" {
					foundTools = true
					break
				}
			}
			if !foundTools {
				parsed.Capabilities = append(parsed.Capabilities, "tools")
			}
		}
	}

	parsed.Architecture = inferPattern(modelID, architecturePatterns, orderedArchKeys)
	if parsed.Architecture == "unknown" {
		parsed.Architecture = inferPattern(parsed.CmdAlias, architecturePatterns, orderedArchKeys)
	}
	if parsed.Architecture == "unknown" {
		parsed.Architecture = inferPattern(parsed.CmdModelPath, architecturePatterns, orderedArchKeys)
	}

	parsed.Family = inferFamilyFromName(modelID, parsed.Architecture)
	if parsed.Family == "unknown" {
		parsed.Family = inferFamilyFromName(parsed.CmdAlias, parsed.Architecture)
	}
	if parsed.Family == "unknown" {
		parsed.Family = inferFamilyFromName(parsed.CmdModelPath, parsed.Architecture)
	}

	parsed.ParameterSize = inferParameterSizeFromName(modelID)
	if parsed.ParameterSize == "unknown" {
		parsed.ParameterSize = inferParameterSizeFromName(parsed.CmdAlias)
	}
	if parsed.ParameterSize == "unknown" {
		parsed.ParameterSize = inferParameterSizeFromName(parsed.CmdModelPath)
	}

	parsed.QuantizationLevel = inferQuantizationLevelFromName(modelID)
	if parsed.QuantizationLevel == "unknown" {
		parsed.QuantizationLevel = inferQuantizationLevelFromName(parsed.CmdAlias)
	}
	if parsed.QuantizationLevel == "unknown" {
		parsed.QuantizationLevel = inferQuantizationLevelFromName(parsed.CmdModelPath)
	}

	return parsed
}
