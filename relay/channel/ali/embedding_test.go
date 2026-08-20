package ali

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/QuantumNous/new-api/common"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relay/constant"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAliMultimodalEmbeddingRequestURL(t *testing.T) {
	tests := []struct {
		name  string
		model string
		want  string
	}{
		{
			name:  "native multimodal embedding",
			model: "qwen3-vl-embedding",
			want:  "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
		},
		{
			name:  "compatible text embedding",
			model: "text-embedding-v1",
			want:  "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := &relaycommon.RelayInfo{
				RelayMode: constant.RelayModeEmbeddings,
				ChannelMeta: &relaycommon.ChannelMeta{
					ChannelBaseUrl:    "https://dashscope.aliyuncs.com",
					UpstreamModelName: tt.model,
				},
			}

			got, err := (&Adaptor{}).GetRequestURL(info)
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestAliMultimodalEmbeddingModelList(t *testing.T) {
	assert.True(t, isAliMultimodalEmbeddingModel("qwen3-vl-embedding"))
	assert.False(t, isAliMultimodalEmbeddingModel("text-embedding-v1"))
}

func TestAliMultimodalEmbeddingResponsePassesThrough(t *testing.T) {
	upstreamBody := []byte(`{
		"output":{"embeddings":[
			{"embedding":[0.1,0.2]}
		]},
		"usage":{"input_tokens":43,"image_tokens":1247,"total_tokens":1290},
		"request_id":"request-1"
	}`)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewReader(upstreamBody)),
	}
	info := &relaycommon.RelayInfo{
		RelayMode: constant.RelayModeEmbeddings,
		ChannelMeta: &relaycommon.ChannelMeta{
			UpstreamModelName: "qwen3-vl-embedding",
		},
	}

	usageValue, apiError := (&Adaptor{}).DoResponse(c, response, info)
	require.Nil(t, apiError)
	usage, ok := usageValue.(*dto.Usage)
	require.True(t, ok)
	assert.Equal(t, 1290, usage.PromptTokens)
	assert.Equal(t, 0, usage.CompletionTokens)
	assert.Equal(t, 1290, usage.TotalTokens)
	assert.Equal(t, 43, usage.PromptTokensDetails.TextTokens)
	assert.Equal(t, 1247, usage.PromptTokensDetails.ImageTokens)

	assert.JSONEq(t, string(upstreamBody), recorder.Body.String())
}

func TestAliMultimodalEmbeddingResponseError(t *testing.T) {
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(bytes.NewBufferString(
			`{"code":"InvalidParameter","message":"invalid multimodal input","request_id":"request-2"}`,
		)),
	}
	info := &relaycommon.RelayInfo{
		RelayMode: constant.RelayModeEmbeddings,
		ChannelMeta: &relaycommon.ChannelMeta{
			UpstreamModelName: "qwen3-vl-embedding",
		},
	}

	usage, apiError := (&Adaptor{}).DoResponse(c, response, info)
	assert.Nil(t, usage)
	require.NotNil(t, apiError)
	assert.Contains(t, apiError.Error(), "invalid multimodal input")
}

func TestAliMultimodalEmbeddingRejectsInvalidUsage(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantError string
	}{
		{
			name:      "negative visual tokens",
			body:      `{"output":{"embeddings":[{"index":0,"embedding":[0.1]}]},"usage":{"input_tokens":1,"image_tokens":-1,"total_tokens":0}}`,
			wantError: "usage",
		},
		{
			name:      "prompt total exceeds billing token bound",
			body:      `{"output":{"embeddings":[{"index":0,"embedding":[0.1]}]},"usage":{"input_tokens":2147483647,"image_tokens":1}}`,
			wantError: "usage",
		},
		{
			name:      "prompt total exceeds int32 billing token bound",
			body:      `{"output":{"embeddings":[{"index":0,"embedding":[0.1]}]},"usage":{"input_tokens":2147483648,"image_tokens":0}}`,
			wantError: "usage",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(recorder)
			response := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewBufferString(tt.body)),
			}
			info := &relaycommon.RelayInfo{
				RelayMode: constant.RelayModeEmbeddings,
				ChannelMeta: &relaycommon.ChannelMeta{
					UpstreamModelName: "qwen3-vl-embedding",
				},
			}

			usage, apiError := (&Adaptor{}).DoResponse(c, response, info)
			assert.Nil(t, usage)
			require.NotNil(t, apiError)
			assert.Contains(t, apiError.Error(), tt.wantError)
		})
	}
}

func TestAliMultimodalEmbeddingConversion(t *testing.T) {
	rawRequest := []byte(`{
		"model":"public-embedding-model",
		"input":{"contents":[
			{"text":"商品描述文本"},
			{"image":"https://example.com/product.png"},
			{"video":"https://example.com/product.mp4"}
		]},
		"parameters":{"enable_fusion":false,"fps":0,"instruct":"Represent the product"}
	}`)
	request := &dto.EmbeddingRequest{}
	require.NoError(t, common.Unmarshal(rawRequest, request))

	requestCopy, err := common.DeepCopy(request)
	require.NoError(t, err)
	requestCopy.Model = "qwen3-vl-embedding"

	convertedValue, err := (&Adaptor{}).ConvertEmbeddingRequest(nil, nil, *requestCopy)
	require.NoError(t, err)
	converted, ok := convertedValue.(*AliEmbeddingRequest)
	require.True(t, ok)

	assert.Equal(t, "qwen3-vl-embedding", converted.Model)
	require.Len(t, converted.Input.Contents, 3)
	assert.Equal(t, "商品描述文本", converted.Input.Contents[0].Text)
	assert.Equal(t, "https://example.com/product.png", converted.Input.Contents[1].Image)
	assert.Equal(t, "https://example.com/product.mp4", converted.Input.Contents[2].Video)
	require.NotNil(t, converted.Parameters)
	require.NotNil(t, converted.Parameters.EnableFusion)
	assert.False(t, *converted.Parameters.EnableFusion)
	require.NotNil(t, converted.Parameters.FPS)
	assert.Equal(t, float64(0), *converted.Parameters.FPS)
	assert.Equal(t, "Represent the product", converted.Parameters.Instruct)
}
