from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RDHParams:
    # 병합 규칙: active > A, (mu_P < mu0 또는 mu_P가 NaN), e = min(s_in, s_out) > tau (tau >= 1.0이면 e == 1.0)
    A: int = 5
    mu0: float | None = 0.61       # None이면 mu 조건을 쓰지 않음
    tau: float = 0.05
    bucket: int = 1200             # active를 세는 버킷 크기 (블록)
    flash_max: int = 100           # 이보다 짧은 위임은 flash (블록)
    rho: float = 0.05              # 지속 시간 로그 빈의 상대 폭
    k: int = 3                     # provider 자기 메뉴를 이루는 상위 빈 수
    min_n: int = 50                # flash 아닌 짝이 이보다 적으면 mu_P = NaN

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class ComponentMap:
    # 크기 2 이상 컴포넌트에 속한 주소마다 한 행. (comp_id, addr) 순 정렬.
    addr: np.ndarray               # int64, 원래 주소 id
    rep: np.ndarray                # int64, 컴포넌트 대표 주소 (컴포넌트 내 최소 주소)
    comp_id: np.ndarray            # int64, rep 오름차순으로 0부터
    comp_size: np.ndarray          # int64, 그 주소가 속한 컴포넌트의 노드 수
    merged_pairs: pd.DataFrame     # src, dst (원래 주소), s_in, s_out, active, mu_P
    params: RDHParams
    start: int
    end: int
    summary: dict = field(default_factory=dict)

    def to_rep(self, addr: np.ndarray) -> np.ndarray:
        """주소를 대표 주소로 바꾼다. 맵에 없는 주소는 그대로 둔다."""
        addr = np.asarray(addr, dtype=np.int64)
        out = addr.copy()
        if self.addr.size == 0:
            return out
        order = np.argsort(self.addr, kind="stable")
        sorted_addr, sorted_rep = self.addr[order], self.rep[order]
        pos = np.minimum(np.searchsorted(sorted_addr, addr), sorted_addr.size - 1)
        hit = sorted_addr[pos] == addr
        out[hit] = sorted_rep[pos[hit]]
        return out
