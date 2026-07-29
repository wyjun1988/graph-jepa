from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.market_data import select_universe
from stock_v2.qwen_events import QwenEventExtractor


if __name__ == "__main__":
    universe = [code for code, _name in select_universe(28)]
    extractor = QwenEventExtractor()
    event = extractor.extract_one(
        title="HBM 수요 급증에 반도체 장비주 강세",
        summary="AI 서버 투자 확대로 HBM 공급망 기업의 실적 기대감이 커지고 있다.",
        universe=universe,
    )
    print(event)
