# Live MMLU Validation Summary — GPT-4o

Real OpenAI API evaluation, gpt-4o-2024-05-13, June 2025.

## Configuration
- Samples: 100 MMLU questions, stratified, seed=42
- Subjects: 41 of 57
- Temperature: 0.0, max_tokens: 8
- API errors (HTTP 429 rate limit): 4

## Results

| Convention | n | Accuracy | 95% CI | vs Calibrated (0.887) |
|---|---|---|---|---|
| Conservative (429=wrong) | 100 | 0.8900 | [0.830, 0.950] | +0.3pp, within CI |
| Successful only | 96 | 0.9271 | [0.875, 0.979] | +4.0pp, within CI |

## Latency (successful calls, retry artifacts excluded)

- p50: 467ms
- p90: 868ms
- p99: 2373ms

## Cost

- Input tokens: 16845, Output tokens: 124
- Total: $0.0861
- Cost per 1M tokens: $5.07

## Conclusion

The calibrated MMLU value (0.887) falls within the live 95 percent CI under both reporting conventions, with point-estimate divergence of just 0.3pp under the conservative convention. This empirically validates the calibrated benchmark distributions.
