# Quantara

Neural-network quantization toolkit. RTN, GPTQ, AWQ and SmoothQuant, quantization-aware training, and layer-by-layer quantization of Llama and Qwen2 checkpoints.

## Setup

```bash
pip install -e ".[llm]"
```

## Usage

```bash
quantara demo
quantara llm quantize /path/to/model out/q --method gptq --calib-files "calib/*.py" --val-files "heldout/*.py"
quantara llm eval out/q --reference /path/to/model --files "heldout/*.py"
```

```python
import quantara as qr

cfg = qr.PTQConfig(weight=qr.QuantConfig(bits=4, group_size=128), method="gptq")
qmodel, report = qr.quantize_model(model, cfg, calib=batches)
```

GPTQ needs more calibration tokens than its widest layer has inputs. With fewer it does worse than RTN.

## Tech Stack

- Python 3.9+, NumPy
- tokenizers (optional, for the `llm` commands)

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
