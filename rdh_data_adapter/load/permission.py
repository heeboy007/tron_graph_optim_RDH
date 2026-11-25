"""Permission and signer records read from DuckDB (inputs of the F2 controller-key filter, rdh.models.filter).

Tables (addresses are integer address ids):
  perm_updates          type-46 AccountPermissionUpdate: owner_address, block_number, tx_id, contract_ret, owner_json, actives_json
  account_create        type-0 AccountCreate: account_address, block_number, contract_ret
  resource_tx           signer columns: permission_id, signer_id (single signature), n_signers
  resource_tx_multisig  one row per signer of a multi-signature resource transaction: tx_id, block_number, signer_idx, signer_id

perm_updates and account_create come from a block scan. They live in the database given by RDH_BLOCKSCAN_DIR, or in the main database
(RDH_DUCKDB_DIR) when that variable is not set (the replica database holds every table).
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from rdh_data_adapter.load.connection import get_duckdb_connection
from rdh_data_adapter.util.environment import get_blockscan_path, get_db_path

MEMORY_LIMIT, THREADS = "12GB", 8

_SQL_MULTISIG = """CREATE TEMP TABLE ms AS
SELECT m.tx_id, m.block_number, r.from_addr AS P, r.type, list(m.signer_id ORDER BY m.signer_idx) AS sids
FROM resource_tx_multisig m JOIN (SELECT tx_id, block_number, from_addr, type FROM resource_tx WHERE n_signers > 1) r USING (tx_id, block_number)
GROUP BY ALL"""


def first_energy_delegation(lo: int, hi: int, db_path=None) -> pd.DataFrame:
    """Account -> block of its first successful ENERGY delegation in [lo, hi]. Columns a, fd (int64), ordered by a."""
    con = get_duckdb_connection(db_path or get_db_path(), memory_limit=MEMORY_LIMIT, threads=THREADS)
    try:
        d = con.execute("""SELECT from_addr AS a, min(block_number) AS fd FROM resource_tx
WHERE block_number BETWEEN $lo AND $hi AND success AND type = 'delegate' AND res_type = 'ENERGY' GROUP BY 1 ORDER BY 1""",
                        {"lo": lo, "hi": hi}).df()
    finally:
        con.close()
    return d.astype({"a": np.int64, "fd": np.int64})


def _register_ids(con, name, col, ids):
    con.register(name, pd.DataFrame({col: np.asarray(sorted(ids), np.int64)}))


def permission_updates(accounts, lo: int, hi: int, db_path=None):
    """Successful permission updates of the given accounts with block in [lo, hi).
    Returns (active, owner, n_rows): active[a] / owner[a] = [(block, frozenset of external key ids)] in (block, tx_id) order.
    Active keys = keys of every active permission; owner keys = keys of the owner permission; the account itself is not a key."""
    con = get_duckdb_connection(db_path or get_blockscan_path(), memory_limit=MEMORY_LIMIT, threads=THREADS)
    try:
        _register_ids(con, "p_", "owner_address", accounts)
        T = con.execute("SELECT t.* FROM perm_updates t JOIN p_ USING (owner_address) WHERE t.block_number >= $lo AND t.block_number < $hi "
                        "AND t.contract_ret = 'SUCCESS' ORDER BY owner_address, block_number, tx_id", {"lo": lo, "hi": hi}).df()
    finally:
        con.close()
    active, owner = defaultdict(list), defaultdict(list)
    for r in T.itertuples():
        o = int(r.owner_address)
        acts = json.loads(r.actives_json) if isinstance(r.actives_json, str) and r.actives_json else []
        own = json.loads(r.owner_json) if isinstance(r.owner_json, str) and r.owner_json else {}
        active[o].append((int(r.block_number), frozenset(k.get("address") for a_ in acts for k in a_.get("keys", []) if k.get("address") != o)))
        owner[o].append((int(r.block_number), frozenset(k["address"] for k in own.get("keys", []) if k["address"] != o)))
    return dict(active), dict(owner), int(len(T))


def account_created(accounts, db_path=None) -> dict:
    """Account -> block of its successful creation (AccountCreate), for the given accounts that have one."""
    con = get_duckdb_connection(db_path or get_blockscan_path(), memory_limit=MEMORY_LIMIT, threads=THREADS)
    try:
        _register_ids(con, "p_", "account_address", accounts)
        C = con.execute("SELECT account_address, min(block_number) b FROM account_create JOIN p_ USING (account_address) "
                        "WHERE contract_ret = 'SUCCESS' GROUP BY 1").df()
    finally:
        con.close()
    return dict(zip(C["account_address"].astype(np.int64), C["b"].astype(int)))


def signing_evidence(accounts, lo: int, hi: int, db_path=None) -> pd.DataFrame:
    """Keys other than P that signed a successful delegate / undelegate of P (any resource, single or multi signature) with block in
    [lo, hi]. Columns P, s, bmin (first such block), ordered by P, s."""
    con = get_duckdb_connection(db_path or get_db_path(), memory_limit=MEMORY_LIMIT, threads=THREADS)
    try:
        con.execute("SET preserve_insertion_order = false")
        con.execute(_SQL_MULTISIG)
        _register_ids(con, "pv", "P", accounts)
        W = f"r.block_number BETWEEN {int(lo)} AND {int(hi)}"
        ev = con.execute(f"""WITH x AS (
  SELECT r.from_addr AS P, r.signer_id AS s, r.block_number AS b FROM resource_tx r
  WHERE {W} AND r.success AND r.type IN ('delegate', 'undelegate') AND r.n_signers = 1 AND r.signer_id <> r.from_addr
  UNION ALL
  SELECT ms.P, unnest(ms.sids), ms.block_number FROM ms
  JOIN (SELECT DISTINCT tx_id, block_number FROM resource_tx r WHERE {W} AND r.n_signers > 1 AND r.success AND r.type IN ('delegate', 'undelegate')) r USING (tx_id, block_number))
SELECT x.P, x.s, min(x.b) AS bmin FROM x JOIN pv USING (P) WHERE x.s <> x.P GROUP BY ALL ORDER BY 1, 2""").df()
    finally:
        con.close()
    return ev.astype({"P": np.int64, "s": np.int64, "bmin": np.int64})


def delegation_counts(lo: int, hi: int, wsz: int, db_path=None) -> pd.DataFrame:
    """Successful delegate (all / ENERGY) and undelegate counts per account and window (window = block // wsz) for blocks in [lo, hi].
    Columns w, P, n_d, n_d_energy, n_u (int64), ordered by w, P. Input of the c decision rule."""
    con = get_duckdb_connection(db_path or get_db_path(), memory_limit=MEMORY_LIMIT, threads=THREADS)
    try:
        d = con.execute(f"""SELECT (block_number // {int(wsz)})::INT AS w, from_addr AS P, count(*) FILTER (WHERE type = 'delegate') AS n_d,
  count(*) FILTER (WHERE type = 'delegate' AND res_type = 'ENERGY') AS n_d_energy, count(*) FILTER (WHERE type = 'undelegate') AS n_u
FROM resource_tx WHERE block_number BETWEEN $lo AND $hi AND success AND type IN ('delegate', 'undelegate')
GROUP BY 1, 2 ORDER BY 1, 2""", {"lo": lo, "hi": hi}).df()
    finally:
        con.close()
    return d.astype({"w": np.int64, "P": np.int64, "n_d": np.int64, "n_d_energy": np.int64, "n_u": np.int64})
