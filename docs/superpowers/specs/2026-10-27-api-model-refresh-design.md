# API Model Refresh

## Intent

Refresh supported cloud model choices and remove retirement risks without
changing local model providers, scoring models, credentials, or saved user
overrides. The user selected balanced cost and quality: stable Flash/common
tiers by default, with flagship/high-speed variants available explicitly.

## Approach

Keep the existing provider and GUI architecture. Update CLI, GUI, provider
fallbacks, and the tracked PowerShell launcher together. Do not introduce live
model discovery or automatically rewrite user configuration. Those alternatives
would add network dependencies or silently change paid workloads.

## Defaults and Options

- Gemini: gemini-3.8-flash.
- Qwen: qwen3.8-flash; retain Max and Omni options.
- MiniMax API and Code: MiniMax-M3.1-Flash; keep M3 and explicit legacy selections.
- Grok Build: grok-4.7, regular tier; keep 4.6 and 4.5 selectable.
- Ark: doubao-seed-2-1-lite-260915; retain Pro and add Turbo; omit the retiring
  doubao-seed-2-0-pro-260215 from suggested choices, but accept explicit IDs.
- GLM: glm-5.3-flash; retain the other existing multimodal choices.
- Step and MiMo: retain step-3.7-flash and mimo-v2.6-flash.
- Mistral: mistral-medium-3-5, a stable multimodal common tier.
- Kimi: add verified K3.1 Pro IDs for each endpoint; retain the existing Kimi
  defaults to avoid increasing subscription quota use without a benchmark.

## Request Compatibility

MiniMax M3.1 forces separated reasoning, uses a configurable reasoning effort
(medium by default), and sends max_completion_tokens instead of max_tokens.
Do not send unsupported sampling controls for M3-family requests. Preserve M3
and legacy request behavior where not contradicted by the documented contract.
Never discard answer content when a stream delta also carries reasoning.
Handle empty/truncated outputs explicitly rather than saving an empty caption.

Kimi K3.1 IDs use the existing low/high/max effort path instead of the legacy
thinking toggle, so stale disabled-thinking settings cannot invalidate requests.
GLM request parameters must be checked against the newly selected default.

## Verification

Test real parser/GUI argument construction, provider request payloads using
mocked network clients, reasoning/answer separation, legacy overrides, and
default consistency. Run the focused tests and the full offline test suite.
No paid requests or model downloads. Account-specific availability remains
unverified until the user runs an actual request.

## References

- https://platform.minimax.io/docs/api-reference/text-openai-api
- https://www.kimi.com/code/docs/en/kimi-code/models.html
- https://docs.x.ai/developers/grok-4-7
- https://docs.volcengine.com/docs/ark/model-list?lang=en
- https://help.aliyun.com/zh/model-studio/vision-model
- https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash
