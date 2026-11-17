from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from quantara._version import __version__
from quantara.cli.llm import cmd_llm_eval, cmd_llm_info, cmd_llm_quantize
from quantara.cli.tensors import (
    _add_quant_args,
    cmd_bench,
    cmd_demo,
    cmd_dequantize,
    cmd_formats,
    cmd_inspect,
    cmd_quantize,
)
from quantara.core.formats import FORMATS
from quantara.errors import QuantaraError

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quantara", description="Neural-network quantization toolkit"
    )
    p.add_argument("--version", action="version", version=f"quantara {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("formats", help="list formats and methods").set_defaults(fn=cmd_formats)

    d = sub.add_parser("demo", help="train a small model and compare RTN, GPTQ, AWQ and QAT")
    d.add_argument("--epochs", type=int, default=40)
    d.add_argument("--output", help="save the int2 QAT model as .qnt")
    d.set_defaults(fn=cmd_demo)

    b = sub.add_parser("bench", help="benchmark number formats on a synthetic matrix")
    b.add_argument("--size", type=int, default=1024)
    b.add_argument("--group-size", type=int, default=128)
    b.set_defaults(fn=cmd_bench)

    q = sub.add_parser("quantize", help="quantize a .npz of weights into a .qnt archive")
    q.add_argument("input"), q.add_argument("output")
    q.add_argument("--min-numel", type=int, default=1024, help="smaller tensors stay in float")
    _add_quant_args(q)
    q.set_defaults(fn=cmd_quantize)

    i = sub.add_parser("inspect", help="show the contents of a .qnt archive")
    i.add_argument("input")
    i.set_defaults(fn=cmd_inspect)

    dq = sub.add_parser("dequantize", help="convert a tensor .qnt back to a float32 .npz")
    dq.add_argument("input"), dq.add_argument("output")
    dq.set_defaults(fn=cmd_dequantize)

    llm = sub.add_parser("llm", help="quantize and evaluate LLMs (safetensors) layer by layer")
    ls = llm.add_subparsers(dest="llm_cmd", required=True)
    li = ls.add_parser("info", help="architecture and size of a checkpoint or Quantara output")
    li.add_argument("model")
    li.set_defaults(fn=cmd_llm_info)

    lq = ls.add_parser("quantize", help="quantize a Hugging Face checkpoint layer by layer")
    lq.add_argument("model", help="folder with config.json, *.safetensors and tokenizer.json")
    lq.add_argument("output")
    lq.add_argument("--method", choices=("rtn", "gptq", "awq", "smoothquant"), default="gptq")
    lq.add_argument("--bits", type=int, default=4)
    lq.add_argument("--format", choices=FORMATS, default="int")
    lq.add_argument("--group-size", type=int, default=128, help="0 = per channel")
    lq.add_argument("--asymmetric", action="store_true")
    lq.add_argument("--observer", choices=("minmax", "percentile", "mse"), default="minmax")
    lq.add_argument("--act-bits", type=int, default=0, help="dynamic int-N activations (0 = float)")
    lq.add_argument("--embed-bits", type=int, choices=(4, 8, 16), default=8)
    lq.add_argument("--head-bits", type=int, choices=(4, 8, 16), default=8)
    lq.add_argument("--calib-files", nargs="+", required=True, help="calibration text files or globs")
    lq.add_argument("--calib-seqs", type=int, default=16)
    lq.add_argument(
        "--val-files", nargs="+", help="text not used for calibration; reports held-out error per layer"
    )
    lq.add_argument("--val-seqs", type=int, default=8)
    lq.add_argument("--seq-len", type=int, default=128)
    lq.add_argument("--calib-rows", type=int, default=1024, help="activation rows kept for AWQ")
    lq.add_argument("--batch-size", type=int, default=4)
    lq.add_argument("--max-layers", type=int, help="quantize only the first N layers")
    lq.add_argument("--act-order", action="store_true", help="GPTQ act_order")
    lq.add_argument("--damp", type=float, default=0.01, help="GPTQ percdamp")
    lq.add_argument("--awq-grid", type=int, default=20)
    lq.add_argument("--sequential", action="store_true", help="GPTQ recalibrates per sublayer (slower)")
    lq.add_argument("--no-resume", action="store_true")
    lq.set_defaults(fn=cmd_llm_quantize)

    le = ls.add_parser("eval", help="perplexity, and comparison with a reference, on text files")
    le.add_argument("model")
    le.add_argument("--reference", help="float checkpoint (or another output) for KL and top-1")
    le.add_argument("--files", nargs="+", required=True)
    le.add_argument("--seqs", type=int, default=8)
    le.add_argument("--seq-len", type=int, default=128)
    le.add_argument("--batch-size", type=int, default=4)
    le.add_argument("--max-layers", type=int)
    le.add_argument("--json", help="write the result as JSON")
    le.set_defaults(fn=cmd_llm_eval)
    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "cmd", "") == "llm" else logging.WARNING,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return args.fn(args)
    except (ValueError, OSError, KeyError, QuantaraError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
