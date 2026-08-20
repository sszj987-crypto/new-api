package dto

import (
	"testing"

	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestEmbeddingRequestKeepsAliParametersInternal(t *testing.T) {
	raw := []byte(`{
		"model":"qwen3-vl-embedding",
		"input":{"contents":[{"text":"商品描述文本"}]},
		"parameters":{"enable_fusion":false,"dimension":1024,"fps":0}
	}`)

	var request EmbeddingRequest
	require.NoError(t, kitutil.Unmarshal(raw, &request))
	assert.JSONEq(t, `{"enable_fusion":false,"dimension":1024,"fps":0}`, string(request.Parameters))

	encoded, err := kitutil.Marshal(request)
	require.NoError(t, err)
	assert.False(t, gjson.GetBytes(encoded, "parameters").Exists())
}

func TestEmbeddingRequestParseInput(t *testing.T) {
	tests := []struct {
		name  string
		input any
		want  []string
	}{
		{name: "single string", input: "first", want: []string{"first"}},
		{name: "empty string", input: "", want: []string{""}},
		{name: "string array", input: []any{"first", "second"}, want: []string{"first", "second"}},
		{
			name: "multimodal string values",
			input: map[string]any{
				"contents": []any{
					map[string]any{"text": "商品描述文本"},
					map[string]any{"image": "https://example.com/product.png"},
					map[string]any{"video": "https://example.com/product.mp4"},
				},
			},
			want: []string{
				"商品描述文本",
				"https://example.com/product.png",
				"https://example.com/product.mp4",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := EmbeddingRequest{Input: tt.input}
			assert.Equal(t, tt.want, request.ParseInput())
		})
	}
}
