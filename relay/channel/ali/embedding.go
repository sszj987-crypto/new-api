package ali

import (
	"fmt"
	"io"
	"math"
	"net/http"

	"github.com/QuantumNous/new-api/common"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/types"
	"github.com/QuantumNous/new-api/service"
	"github.com/gin-gonic/gin"
)

var aliMultimodalEmbeddingModels = []string{
	"qwen3-vl-embedding",
}

func isAliMultimodalEmbeddingModel(model string) bool {
	for _, candidate := range aliMultimodalEmbeddingModels {
		if model == candidate {
			return true
		}
	}
	return false
}

func requestOpenAI2AliMultimodalEmbedding(request dto.EmbeddingRequest) (*AliEmbeddingRequest, error) {
	converted := &AliEmbeddingRequest{
		Model: request.Model,
	}
	inputData, err := common.Marshal(request.Input)
	if err != nil {
		return nil, err
	}
	if err = common.Unmarshal(inputData, &converted.Input); err != nil {
		return nil, err
	}
	if len(request.Parameters) > 0 {
		converted.Parameters = &AliEmbeddingParameters{}
		if err = common.Unmarshal(request.Parameters, converted.Parameters); err != nil {
			return nil, err
		}
	}
	return converted, nil
}

func aliMultimodalEmbeddingHandler(c *gin.Context, resp *http.Response, info *relaycommon.RelayInfo) (*dto.Usage, *types.NewAPIError) {
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeBadResponseBody, http.StatusBadGateway)
	}
	service.CloseResponseBodyGracefully(resp)

	aliResponse := AliEmbeddingResponse{}
	if err = common.Unmarshal(responseBody, &aliResponse); err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeBadResponseBody, http.StatusBadGateway)
	}
	if aliResponse.Code != "" {
		return nil, types.NewOpenAIError(
			fmt.Errorf("Ali error %s: %s (request_id: %s)", aliResponse.Code, aliResponse.Message, aliResponse.RequestId),
			types.ErrorCodeBadResponseBody,
			http.StatusBadGateway,
		)
	}
	if aliResponse.Usage.InputTokens < 0 || aliResponse.Usage.ImageTokens < 0 {
		return nil, types.NewOpenAIError(
			fmt.Errorf("Ali multimodal embedding usage contains negative token counts"),
			types.ErrorCodeBadResponseBody,
			http.StatusBadGateway,
		)
	}
	promptTokens64 := int64(aliResponse.Usage.InputTokens) + int64(aliResponse.Usage.ImageTokens)
	if promptTokens64 > math.MaxInt32 {
		return nil, types.NewOpenAIError(
			fmt.Errorf("Ali multimodal embedding usage exceeds the billing token limit"),
			types.ErrorCodeBadResponseBody,
			http.StatusBadGateway,
		)
	}
	promptTokens := int(promptTokens64)
	usage := &dto.Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: 0,
		TotalTokens:      promptTokens,
		PromptTokensDetails: dto.InputTokenDetails{
			TextTokens:  aliResponse.Usage.InputTokens,
			ImageTokens: aliResponse.Usage.ImageTokens,
		},
	}

	service.IOCopyBytesGracefully(c, resp, responseBody)
	return usage, nil
}
