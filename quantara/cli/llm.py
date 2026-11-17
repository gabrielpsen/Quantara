from __future__ import annotations

import glob
import json
import os
from typing import List

from quantara.core.formats import QuantConfig
from quantara.core.observers import ActConfig
from quantara.llm.checkpoint import Checkpoint
from quantara.llm.data import load_tokenizer, tokenize_files
from quantara.llm.pipeline import LLMQuantConfig, quantize_llm
from quantara.llm.runtime import CausalLM, compare_logits, next_token_nll
from quantara.llm.transformer import DecoderConfig
from quantara.quantize.config import PTQConfig

def _expand(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for p in patterns:
        files.extend(h for h in sorted(glob.glob(p, recursive=True)) if os.path.isfile(h))
    if not files:
        raise ValueError(f"no files match {patterns}")
    return files

def cmd_llm_info(a) -> int:
    if os.path.exists(os.path.join(a.model, "quantara.json")):
        with open(os.path.join(a.model, "quantara.json"), encoding="utf-8") as f:
            m = json.load(f)
        c = m["config"]
        print(f"Quantara output v{m['quantara_version']}  (source: {m['source']})")
        print(
            f"  method={c['method']}  weights={c['weight']['format']}{c['weight']['bits']} "
            f"g{c['weight']['group_size']}  embed={c['embed_bits']}b  head={c['head_bits']}b"
        )
        fp, q = m["decoder_fp_bytes"], m["decoder_quant_bytes"]
        print(
            f"  decoder layers: {m['layers_quantized']}  "
            f"{fp / 1e9:.2f} GB -> {q / 1e9:.2f} GB ({fp / max(q, 1):.2f}x)"
        )
        print(
            f"  embed {m['io']['embed_bytes'] / 1e6:.0f} MB, head {m['io']['head_bytes'] / 1e6:.0f} MB, "
            f"quantization took {m['seconds'] / 60:.1f} min"
        )
        return 0
    with Checkpoint(a.model) as ck:
        dc = DecoderConfig.from_hf(ck.config)
        print(
            f"{a.model}\n  {ck.config.get('model_type')}  layers={dc.num_layers} hidden={dc.hidden_size} "
            f"heads={dc.num_heads}/{dc.num_kv_heads}kv  ffn={dc.intermediate_size} vocab={dc.vocab_size}"
        )
        print(f"  {ck.num_parameters() / 1e9:.2f}B parameters, {ck.total_bytes() / 1e9:.2f} GB on disk")
    return 0

def cmd_llm_quantize(a) -> int:
    wcfg = QuantConfig(
        bits=a.bits,
        format=a.format,
        symmetric=not a.asymmetric,
        granularity="group" if a.group_size else "channel",
        group_size=a.group_size or 128,
        observer=a.observer,
    )
    act = ActConfig(bits=a.act_bits, dynamic=True) if a.act_bits else None
    ptq = PTQConfig(
        weight=wcfg,
        method=a.method,
        act=act,
        gptq_act_order=a.act_order,
        gptq_percdamp=a.damp,
        awq_grid=a.awq_grid,
        calib_samples=a.calib_rows,
        cache_weights=False,
        sequential=a.sequential,
    )
    cfg = LLMQuantConfig(ptq=ptq, embed_bits=a.embed_bits, head_bits=a.head_bits, batch_size=a.batch_size)
    tok = load_tokenizer(a.model)
    calib = tokenize_files(tok, _expand(a.calib_files), a.seq_len, a.calib_seqs)
    val = tokenize_files(tok, _expand(a.val_files), a.seq_len, a.val_seqs) if a.val_files else None
    m = quantize_llm(
        a.model, a.output, cfg, calib, max_layers=a.max_layers, resume=not a.no_resume, val_tokens=val
    )
    fp, q = m["decoder_fp_bytes"], m["decoder_quant_bytes"]
    print(
        f"\ndone in {m['seconds'] / 60:.1f} min  |  decoder {fp / 1e9:.2f} GB -> {q / 1e9:.2f} GB "
        f"({fp / q:.2f}x)  ->  {a.output}"
    )
    return 0

def cmd_llm_eval(a) -> int:
    files = _expand(a.files)
    with CausalLM(a.model) as lm:
        toks = tokenize_files(lm.tokenizer(), files, a.seq_len, a.seqs)
        if a.max_layers:
            lm.n_layers = min(lm.n_layers, a.max_layers)
        lg = lm.logits(toks, a.batch_size)
    result = {"model": a.model, "tokens": int(toks.size), **next_token_nll(lg, toks)}
    if a.reference:
        with CausalLM(a.reference) as ref:
            if a.max_layers:
                ref.n_layers = min(ref.n_layers, a.max_layers)
            rl = ref.logits(toks, a.batch_size)
        result.update(compare_logits(rl, lg, toks))
        result["reference"] = a.reference
    for k, v in result.items():
        print(f"{k:>24}: {v:.5g}" if isinstance(v, float) else f"{k:>24}: {v}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return 0
