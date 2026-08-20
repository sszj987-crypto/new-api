package dto

import (
	"encoding/json"
	"net/http"
	"strings"

	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
	"github.com/QuantumNous/new-api/relaykit/types"
)

type EmbeddingOptions struct {
	Seed             int      `json:"seed,omitempty"`
	Temperature      *float64 `json:"temperature,omitempty"`
	TopK             int      `json:"top_k,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	NumPredict       int      `json:"num_predict,omitempty"`
	NumCtx           int      `json:"num_ctx,omitempty"`
}

type EmbeddingRequest struct {
	Model            string          `json:"model"`
	Input            any             `json:"input"`
	EncodingFormat   string          `json:"encoding_format,omitempty"`
	Dimensions       *int            `json:"dimensions,omitempty"`
	User             string          `json:"user,omitempty"`
	Seed             *float64        `json:"seed,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	Parameters       json.RawMessage `json:"-"`
}

func (r *EmbeddingRequest) UnmarshalJSON(data []byte) error {
	type alias EmbeddingRequest
	wireRequest := struct {
		*alias
		Parameters json.RawMessage `json:"parameters"`
	}{
		alias: (*alias)(r),
	}
	if err := kitutil.Unmarshal(data, &wireRequest); err != nil {
		return err
	}
	r.Parameters = wireRequest.Parameters
	return nil
}

func (r *EmbeddingRequest) GetTokenCountMeta() *types.TokenCountMeta {
	var texts = make([]string, 0)

	inputs := r.ParseInput()
	for _, input := range inputs {
		texts = append(texts, input)
	}

	return &types.TokenCountMeta{
		CombineText: strings.Join(texts, "\n"),
	}
}

func (r *EmbeddingRequest) IsStream(c *http.Request) bool {
	return false
}

func (r *EmbeddingRequest) SetModelName(modelName string) {
	if modelName != "" {
		r.Model = modelName
	}
}

func (r *EmbeddingRequest) ParseInput() []string {
	if r.Input == nil {
		return make([]string, 0)
	}
	return embeddingStringValues(r.Input)
}

func embeddingStringValues(value any) []string {
	switch value := value.(type) {
	case string:
		return []string{value}
	case map[string]any:
		values := make([]string, 0)
		for _, item := range value {
			values = append(values, embeddingStringValues(item)...)
		}
		return values
	case []any:
		values := make([]string, 0, len(value))
		for _, item := range value {
			values = append(values, embeddingStringValues(item)...)
		}
		return values
	}
	return nil
}

type EmbeddingResponseItem struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type EmbeddingResponse struct {
	Object string                  `json:"object"`
	Data   []EmbeddingResponseItem `json:"data"`
	Model  string                  `json:"model"`
	Usage  `json:"usage"`
}
