"""CPM labels of several chunks, one process per chunk.

    python -m model.ext_features.cpm --chunks=79200000,79400000 --out=dump/v1/cpm [--parallel=3] [--ref-chunk=80000000] [--log2-gamma=-10]
Writes <out>/labels/leiden_cpm_week_<chunk>.parquet (existing files are kept) and <out>/gamma.json.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .cpm import LOG2_GAMMA, REF_CHUNK, gamma_c_of, write_chunk_labels


def _arg(name, default=None):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def main():
    out = Path(_arg("out"))
    chunks = [int(x) for x in _arg("chunks").split(",")]
    ref, log2g = int(_arg("ref-chunk", REF_CHUNK)), float(_arg("log2-gamma", LOG2_GAMMA))
    out.mkdir(parents=True, exist_ok=True)
    gf = out / "gamma.json"
    meta = json.loads(gf.read_text()) if gf.exists() else {}
    if meta.get("ref_chunk") != ref or meta.get("log2_gamma") != log2g:
        meta = {"ref_chunk": ref, "log2_gamma": log2g, "gamma_c": gamma_c_of(ref, log2g)}
        gf.write_text(json.dumps(meta, indent=1))
    with ProcessPoolExecutor(int(_arg("parallel", 3))) as ex:
        for c, msg in zip(chunks, ex.map(write_chunk_labels, chunks, [meta["gamma_c"]] * len(chunks), [out] * len(chunks))):
            print(f"chunk {c}: {msg}", flush=True)


if __name__ == "__main__":
    main()
