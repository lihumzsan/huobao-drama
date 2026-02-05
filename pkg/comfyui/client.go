package comfyui

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client 调用 ComfyUI API 提交工作流并轮询结果（与 file1.html 中 Flux 文生图工作流一致）
type Client struct {
	BaseURL  string
	ClientID string
	HTTP     *http.Client
}

// Params 文生图参数
type Params struct {
	Prompt string
	Width  int // 默认 1080
	Height int // 默认 1920
	Steps  int // 默认 25
	CFG    float64
	Seed   int64
	// 可选：百度翻译 API（工作流含 BaiduTranslateNode 时使用，为空则不走翻译）
	BaiduTranslateAppID  string
	BaiduTranslateAppKey string
}

// Generate 提交工作流并等待完成，返回生成图片的完整 URL（BaseURL + /view?filename=...）
func (c *Client) Generate(p *Params) (imageURL string, err error) {
	if p.Width <= 0 {
		p.Width = 1080
	}
	if p.Height <= 0 {
		p.Height = 1920
	}
	if p.Steps <= 0 {
		p.Steps = 25
	}
	if p.CFG <= 0 {
		p.CFG = 1
	}
	if p.Seed == 0 {
		p.Seed = time.Now().UnixNano() % 100000000000000
	}

	workflow := c.buildWorkflow(p.Prompt, p.Width, p.Height, p.Steps, p.CFG, p.Seed, p.BaiduTranslateAppID, p.BaiduTranslateAppKey)
	baseURL := c.BaseURL
	if baseURL == "" {
		return "", fmt.Errorf("comfyui base_url is required")
	}
	if baseURL[len(baseURL)-1] == '/' {
		baseURL = baseURL[:len(baseURL)-1]
	}
	clientID := c.ClientID
	if clientID == "" {
		clientID = "huobao_drama"
	}

	body, _ := json.Marshal(map[string]interface{}{
		"prompt":    workflow,
		"client_id": clientID,
	})
	req, err := http.NewRequest("POST", baseURL+"/prompt", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	hc := c.HTTP
	if hc == nil {
		hc = &http.Client{Timeout: 30 * time.Second}
	}
	resp, err := hc.Do(req)
	if err != nil {
		return "", fmt.Errorf("comfyui submit: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("comfyui submit %s: %s", resp.Status, string(b))
	}

	var submitResp struct {
		PromptID string `json:"prompt_id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&submitResp); err != nil {
		return "", fmt.Errorf("comfyui decode submit response: %w", err)
	}
	if submitResp.PromptID == "" {
		return "", fmt.Errorf("comfyui no prompt_id in response")
	}

	// 轮询 /history/{prompt_id}
	for i := 0; i < 300; i++ {
		time.Sleep(1 * time.Second)
		histReq, _ := http.NewRequest("GET", baseURL+"/history/"+submitResp.PromptID, nil)
		histResp, err := hc.Do(histReq)
		if err != nil {
			continue
		}
		var history map[string]struct {
			Outputs map[string]struct {
				Images []struct {
					Filename  string `json:"filename"`
					Subfolder string `json:"subfolder"`
					Type      string `json:"type"`
				} `json:"images"`
			} `json:"outputs"`
		}
		_ = json.NewDecoder(histResp.Body).Decode(&history)
		histResp.Body.Close()

		entry, ok := history[submitResp.PromptID]
		if !ok {
			continue
		}
		for _, out := range entry.Outputs {
			if len(out.Images) > 0 {
				img := out.Images[0]
				imageURL = fmt.Sprintf("%s/view?filename=%s&subfolder=%s&type=%s",
					baseURL, img.Filename, img.Subfolder, img.Type)
				return imageURL, nil
			}
		}
	}
	return "", fmt.Errorf("comfyui timeout waiting for result")
}

// buildWorkflow 与 flux.json 一致：含 BaiduTranslateNode(24) -> CLIPTextEncode(21)，其余为 Flux 文生图
func (c *Client) buildWorkflow(prompt string, width, height, steps int, cfg float64, seed int64, baiduAppID, baiduAppKey string) map[string]interface{} {
	// 节点 24：BaiduTranslateNode，输入为 prompt（中译英等），输出给 21
	inputs24 := map[string]interface{}{
		"from_translate": "auto",
		"to_translate":   "en",
		"text":           prompt,
	}
	if baiduAppID != "" && baiduAppKey != "" {
		inputs24["baidu_appid"] = baiduAppID
		inputs24["baidu_appkey"] = baiduAppKey
	}
	node24 := map[string]interface{}{
		"inputs":     inputs24,
		"class_type": "BaiduTranslateNode",
	}

	return map[string]interface{}{
		"4": map[string]interface{}{
			"inputs":     map[string]interface{}{"conditioning": []interface{}{"21", 0}},
			"class_type": "ConditioningZeroOut",
		},
		"5": map[string]interface{}{
			"inputs":     map[string]interface{}{"samples": []interface{}{"15", 0}, "vae": []interface{}{"19", 0}},
			"class_type": "VAEDecode",
		},
		"8": map[string]interface{}{
			"inputs":     map[string]interface{}{"filename_prefix": "comfy_ui_generated", "images": []interface{}{"5", 0}},
			"class_type": "SaveImage",
		},
		"15": map[string]interface{}{
			"inputs": map[string]interface{}{
				"seed": seed, "steps": steps, "cfg": cfg,
				"sampler_name": "euler", "scheduler": "beta", "denoise": 1,
				"model": []interface{}{"17", 0}, "positive": []interface{}{"21", 0},
				"negative": []interface{}{"4", 0}, "latent_image": []interface{}{"20", 0},
			},
			"class_type": "KSampler",
		},
		"17": map[string]interface{}{
			"inputs":     map[string]interface{}{"unet_name": "flux\\flux1-dev.safetensors", "weight_dtype": "fp8_e4m3fn"},
			"class_type": "UNETLoader",
		},
		"18": map[string]interface{}{
			"inputs": map[string]interface{}{
				"clip_name1": "flux\\t5xxl_fp8_e4m3fn.safetensors",
				"clip_name2": "flux\\clip_l.safetensors",
				"type":       "flux", "device": "default",
			},
			"class_type": "DualCLIPLoader",
		},
		"19": map[string]interface{}{
			"inputs":     map[string]interface{}{"vae_name": "flux\\ae.safetensors"},
			"class_type": "VAELoader",
		},
		"20": map[string]interface{}{
			"inputs":     map[string]interface{}{"width": width, "height": height, "batch_size": 1},
			"class_type": "EmptyLatentImage",
		},
		"21": map[string]interface{}{
			"inputs":     map[string]interface{}{"text": []interface{}{"24", 0}, "clip": []interface{}{"18", 0}},
			"class_type": "CLIPTextEncode",
		},
		"24": node24,
	}
}
