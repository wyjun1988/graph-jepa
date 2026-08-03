# 설계 — 수급 랭킹 손실 헤드 (flow-rank, `fr_s`)

2026-08-03 작성. 다음 4090 세션의 실험 슬롯 후보.
hz(지평 헤드)가 접히고 epc(랭킹 압력)가 기각되면서 비어 있는 자리다.

## 1. 왜 — 근거 세 겹

| 사실 | 수치 | 출처 |
|---|---:|---|
| 목표에 값이 있다 | 실제 `t+h` 연기금 흐름의 IC = **+0.134** | `PENSION_FLOW_ALPHA_20260803.md` |
| 그 값은 챔프 신호보다 크다 | 챔프 `pred_return` IC = +0.057 (**2.3배**) | 같은 문서 |
| 현행 예측이 명백히 나쁘다 | 모델 +0.112 vs **persistence +0.210** | 같은 문서 |

현행 모델은 수급을 **MSE 로 복원**하도록만 학습된다(149개 상태피처 중 하나).
MSE 정확도(pooled skill +0.50)가 **횡단면 랭킹**으로 전이되지 않았고, 그 결과
예측이 과거 흐름 쪽으로 쏠려 음수 부호(IC −0.078)를 물려받았다.

문헌도 같은 방향이다 — 상위 바스켓 과제에서 손실함수 선택이 결과를 가른다
(arXiv 2510.14156, 2026-08-02 조사분).

**가설:** 수급 피처에 **횡단면 상관 손실**을 직접 걸면 랭킹 정확도가 오르고,
그 예측이 선택 신호에 더해질 값어치를 갖는다.

## 2. 무엇 — 최소 변경 설계

이미 있는 기계를 재사용한다. 새 헤드·새 아키텍처가 **아니다**.

- `_grouped_correlation_loss(pred, target, valid, graph_index)` — 횡단면 상관 손실.
  `entry_path_correlation_loss` 가 이미 쓰고 있다(`graph_jepa.py:1199`).
- 수급 예측값은 이미 `state_pred` 안에 있다(149 상태피처에 포함).
- 목표값도 이미 `target_batch.node_features` 안에 있다.

즉 **둘을 이어 붙이기만** 하면 된다.

### 변경 지점 (4곳)

| # | 파일:위치 | 내용 |
|---|---|---|
| 1 | `graph_jepa.py:~537` 생성자 | `flow_rank_loss_weight: float = 0.0`, `flow_rank_features: Sequence[str] = ()` |
| 2 | `graph_jepa.py:~771` 인덱스 해석 | `self.flow_feature_indices = [names.index(n) for n in flow_rank_features if n in names]` — `gap_open_feature_index` 와 동일 패턴 |
| 3 | `graph_jepa.py:~1199` 뒤 | `_flow_rank_loss()` 신설 (아래) |
| 4 | `graph_jepa.py:2373, 2643` 합산 | `+ self.flow_rank_loss_weight * flow_rank_loss` |

```python
def _flow_rank_loss(self, state_pred, target_batch, mask):
    """수급 피처의 횡단면 랭킹 손실. MSE 복원과 별도로 순위를 직접 맞춘다."""
    if self.flow_rank_loss_weight <= 0.0 or not self.flow_feature_indices:
        return state_pred.new_tensor(0.0)
    total = state_pred.new_tensor(0.0)
    for idx in self.flow_feature_indices:
        pred = state_pred[:, idx]
        tgt = target_batch.node_features[:, idx]
        valid = (mask[:, idx] & torch.isfinite(pred) & torch.isfinite(tgt))
        total = total + self._grouped_correlation_loss(
            pred, tgt, valid, target_batch.graph_index)
    return total / len(self.flow_feature_indices)
```

`run_real_backtest.py` 에 CLI 2개 추가:
`--flow-rank-loss-weight`(기본 0.0), `--flow-rank-features`(기본
`investor_pension_flow_ratio_1d`). 기본값 0 이므로 **미지정 시 현행과 완전히 동일**하다.

## 3. 실험 행렬

프리픽스 `fr_s`. 베이스라인은 같은 머신의 `ens_s`(머신 짝지음 유지).

| 단계 | 설정 | 런수 |
|---|---|---:|
| S-a 스모크 | `w=0.25`, r5, 시드 17, 3에폭 | 1 |
| S-b 본실험 | `w=0.25`, r5·r4, 시드 3·17·29 | 6 |
| S-c 가중 스윕 | `w∈{0.1, 0.5}`, r5, 시드 3·17·29 | 6 |

**13런.** 4090 실측 25~45분/런, 동시 2 → **3~5시간**. 5폴드 확장은 S-b 통과 후.

가중 시작값 0.25 근거: `epc` 가 같은 계열 손실을 0.05→0.25 로 올렸을 때
h10 IC 회귀가 유의하지 않았다(r1 제외). 즉 0.25 는 주 손실을 망가뜨리지 않는
검증된 크기다.

## 4. 사전등록 판정 (r1~r3 미관측 상태에서 등록)

**게이트 1 — 회귀 금지 (필수).** `paired_variant_report --a ens_s --b fr_s` 에서
h10 IC 가 유의하게 나빠지지 않을 것. 나빠지면 즉시 폐기(epc 와 동일 기준).

**게이트 2 — 예측 품질 (주판정).**
`pension_pred_quality_study` 의 [1] 에서
`corr(fr_s 예측, 실제 t+h 흐름)` 가 **persistence(+0.210)를 넘을 것.**
현행 `ens_s` 는 +0.112 로 미달이다. 이걸 못 넘으면 손실 추가가 목적을 달성 못 한 것.

**게이트 3 — 실현 알파 (본목적).**
`pension_pred_alpha_study` 에서 `pred_pension` 의 IC 가 **양수**이고,
`combo z(ret)+w·z(pension)` 가 `pred_return` 단독(+0.0574)을 넘을 것.

세 게이트를 모두 통과해야 프로덕션 후보다. 게이트 2 만 통과하고 3 이 실패하면
"예측은 좋아졌으나 알파가 아니다" 로 기록하고 닫는다.

## 5. 비용과 위험

**비용:** 13런 3~5시간(4090). 평가·판정은 CPU 로 기존 스크립트 재사용.

**위험 1 — 주 손실 훼손.** 수급 랭킹을 밀다 수익률 예측이 나빠질 수 있다.
게이트 1 이 이걸 잡는다. `epc` 가 r1 에서 유의하게 나빴던 전례가 있어 실재하는 위험이다.

**위험 2 — 예측이 좋아져도 알파가 없을 수 있다.** persistence 가 맞히는 성분
(직전 흐름)은 IC −0.055 다. 랭킹 손실이 그 성분만 더 잘 맞히면 알파는 오히려
악화된다. 게이트 3 이 이걸 잡는다.

**위험 3 — 2폴드 함정.** SL−5% 가 정확히 이 자리에서 무너졌다. S-b 는 2폴드이므로
통과해도 **채택이 아니라 5폴드 승급**일 뿐이다.

**중단 기준:** S-a 스모크에서 손실이 발산하거나 h10 IC 가 눈에 띄게 무너지면
본실험 생략. `4090_all.sh` 의 hz 스모크 패턴을 그대로 쓴다(실패 시 학습 로그 tail 출력).

## 6. 실행 순서

```
git pull
# 1) 코드 변경 4곳 + CLI 2개
# 2) 스모크
PREFIX=fr_s EXTRA="--flow-rank-loss-weight 0.25 --epochs 3 --checkpoint-epochs 2" \
  bash scripts/seed_queue_v2.sh 1 r5 17
# 3) 본실험
for F in r5 r4; do
  PREFIX=fr_s EXTRA="--flow-rank-loss-weight 0.25" \
    bash scripts/seed_queue_v2.sh 2 $F 3 17 29
done
# 4) 판정 (CPU)
python scripts/paired_variant_report.py --a ens_s --b fr_s --seeds 3 17 29 --fold r5
python scripts/pension_pred_quality_study.py --dirs <fr_s eval dirs> --horizon 10
python scripts/pension_pred_alpha_study.py  --dirs <fr_s eval dirs> --hold 10
```

**주의:** 게이트 2·3 을 재려면 평가 CSV 에 수급 열이 있어야 한다.
`evaluate_node_prediction.py` 의 `optional_forecast_names` 추가분(2026-08-03,
커밋 695ad92)이 이미 들어가 있으므로 `git pull` 만 하면 된다.

## 7. 이 설계가 기각되는 조건

수급 랭킹 손실이 게이트 1 을 못 넘으면(주 손실 훼손) 이 방향은 닫는다.
`epc` 가 같은 이유로 기각됐고, 두 번째 같은 실패면 "이 인코더에 보조 랭킹 손실을
얹는 것" 자체가 안 되는 것으로 본다.
