## DesAdapter

`Des-HyperNEAT-Python` 코어를 건드리지 않고, `k_flower_card`용 DES 학습/듀얼/어댑터 코드를 분리해서 두는 폴더다.

의도된 배치:
- `python/`: matgo 학습/평가/런타임 export
- `js/`: 브라우저/엔진 쪽 matgo adapter
- `configs/`: DES matgo 전용 설정

코어는 여기 넣지 않는다.

현재 엔트리:
- `python/train_des_matgo.py`
- 기본 runtime: `configs/runtime_phase1.json`
