"""사다리와 D+20 의 Sharpe 가 같아지는 왕복 비용(손익분기).

실주문이 0이라 41~108bp 는 전부 추정치다(docs §8-9). 그래서 "실비용이 얼마
이상이면 결론이 바뀌는가"가 판단에 직결된다. 손익분기가 현행 가정(41bp)보다
훨씬 낮으면, 비용 추정이 틀려도 결론은 안 바뀐다는 뜻이다.
"""
import sys, math
sys.path.insert(0, "scripts")
from exit_policy_report import (load_prices, load_index, load_picks, select,
    make_benchmark, policy_returns, POLICY_HOLD, TRADING_DAYS, FOLDS, TOP_N)

seeds = [3, 5, 11, 17, 23]
L = {}
for s in seeds:
    d = load_picks("ens_s%d" % s, FOLDS["r5"])
    if d:
        L[s] = d
panel = load_prices()
idx = load_index()
sk = sorted(L)
dates = sorted(set.intersection(*(set(L[s]) for s in sk)))
ub = {d: sorted(L[sk[0]][d], key=lambda t: L[sk[0]][d][t][1], reverse=True)[:TOP_N]
      for d in dates}
bench = make_benchmark("universe", panel, idx, ub)
ens = {}
for d in dates:
    tk = set.intersection(*(set(L[s][d]) for s in sk))
    base = L[sk[0]][d]
    ens[d] = {t: (sum(L[s][d][t][0] for s in sk) / len(sk), base[t][1]) for t in tk}


def stats(daily, h):
    v = [x for _, x in sorted(daily.items())]
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    return m, sd, TRADING_DAYS / h


print("시드 %s / 폴드 r5 / 공통 %d일" % (sk, len(dates)))
print("%8s %5s %13s" % ("변형", "편입", "손익분기bp"))
print("-" * 30)
for K in (5, 20):
    variants = [("앙상블", {d: select({d: ens[d]}, d, K) for d in dates})]
    variants += [("seed %d" % s, {d: select(L[s], d, K) for d in dates}) for s in sk]
    for label, pb in variants:
        pol = policy_returns(panel, bench, pb)
        mL, sL, tL = stats(pol["사다리(1,2,3,5,10)"], POLICY_HOLD["사다리(1,2,3,5,10)"])
        m2, s2, t2 = stats(pol["단일 D+20"], POLICY_HOLD["단일 D+20"])
        # Sharpe(c) = (m - c) * turns / (sd * sqrt(turns)) = (m - c) * sqrt(turns) / sd
        kL = math.sqrt(tL) / sL
        k2 = math.sqrt(t2) / s2
        # kL*(mL - c) = k2*(m2 - c)  ->  c = (kL*mL - k2*m2) / (kL - k2)
        c = (kL * mL - k2 * m2) / (kL - k2) if abs(kL - k2) > 1e-12 else float("nan")
        print("%8s %5d %13.1f" % (label, K, c * 10000))
print()
print("해석: 실제 왕복비용이 이 값보다 크면 D+20 이 사다리보다 낫다.")
print("현행 가정 41bp, 호가단위 하한 26bp (docs §8-5).")
