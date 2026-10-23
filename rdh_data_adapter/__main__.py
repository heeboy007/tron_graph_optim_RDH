"""
CLI: python -m rdh_data_adapter component-map --start 80000000 --end 80200000 --out path [--tau --mu0 --A ...]

rdh는 rdh_data_adapter(DB)를 import하지 않으므로 DB를 읽는 CLI는 이 패키지에 둔다.
"""
import argparse
import json

from rdh import RDHParams, save_as_pq
from rdh_data_adapter.helper import build_component_map
from rdh_data_adapter.load.resource_graph import PAIRING_MEMORY_LIMIT

def _mu0(value: str):
    return None if value.lower() == "none" else float(value)

def main(argv=None):
    d = RDHParams()
    parser = argparse.ArgumentParser(prog="python -m rdh_data_adapter")
    sub = parser.add_subparsers(dest="command", required=True)
    cm = sub.add_parser("component-map", help="effect size 문턱으로 컴포넌트 맵을 만들어 저장한다")
    cm.add_argument("--start", type=int, required=True)
    cm.add_argument("--end", type=int, required=True)
    cm.add_argument("--out", required=True, help="저장할 폴더 (parquet + meta.json)")
    cm.add_argument("--db-path", default=None, help="tron.db 경로 (없으면 RDH_DUCKDB_DIR)")
    cm.add_argument("--memory-limit", default=PAIRING_MEMORY_LIMIT, help="DuckDB memory_limit (예: 1GB, 4GB)")
    cm.add_argument("--A", type=int, default=d.A)
    cm.add_argument("--mu0", type=_mu0, default=d.mu0, help="'none'이면 mu 조건을 쓰지 않음")
    cm.add_argument("--tau", type=float, default=d.tau)
    cm.add_argument("--bucket", type=int, default=d.bucket)
    cm.add_argument("--flash-max", type=int, default=d.flash_max)
    cm.add_argument("--rho", type=float, default=d.rho)
    cm.add_argument("--k", type=int, default=d.k)
    cm.add_argument("--min-n", type=int, default=d.min_n)
    args = parser.parse_args(argv)

    params = RDHParams(A=args.A, mu0=args.mu0, tau=args.tau, bucket=args.bucket,
                       flash_max=args.flash_max, rho=args.rho, k=args.k, min_n=args.min_n)
    cmap = build_component_map(args.start, args.end, params, db_path=args.db_path, memory_limit=args.memory_limit)
    out = save_as_pq(args.out, cmap)
    print(json.dumps({"out": str(out), **cmap.summary}, ensure_ascii=False))

if __name__ == "__main__":
    main()
