# 1. 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 기간 | 2021.10.25 – 2021.10.29 (1주) |
| 팀원 | 안희진 김소연 김지수 정영빈 |
| 주제 | Yelp 리뷰 긍/부정 이진분류 모델 설계 및 성능 최적화 |
| 내용 | Yelp 리뷰 텍스트를 기반으로 긍/부정 감정을 이진 분류할 수 있는 고성능 모델을 개발하는 것을 목표로 하였습니다. Huggingface Transformers 프레임워크를 활용해 다양한 Transformer 기반 모델(BERT, RoBERTa, ALBERT, SCI-BERT)을 실험하였으며, SCI-BERT 모델이 테스트 스코어 0.987로 가장 우수한 성능을 보였습니다. 최종적으로 Hard-voting 앙상블 기법을 적용하여 0.992의 높은 성능을 달성하였습니다. |
| 결과 | - 정확도 0.992 <br>- 교육과정 캐글 대회에서 8팀 중 1위 차지 |
| 기술스택 | 프레임워크 및 라이브러리: Pytorch, Huggingface Transformers, Tensorboard<br>데이터 처리 및 분석: Numpy, Pandas, Matplotlib<br>모델 및 기법:<br>- Transformer 기반 모델: BERT, RoBERTa, ALBERT, SCI-BERT<br>- 기법: 앙상블(Hard-voting)<br>협업 및 기타도구: Github, Kaggle, Jupyter Notebook |
| 코드URL | [https://github.com/kkogggokk/TextClassification](https://github.com/kkogggokk/TextClassification)<br> [https://](https://github.com/kkogggokk/TextClassification)[www.kaggle.com/c/goormtextclassificationproject](https://www.kaggle.com/c/goormtextclassificationproject)    |

# 2. 프로젝트 진행 프로세스

## 2.1 베이스라인 분석 및 사전조사

- **Test Data Collate Function 수정**: 데이터 순서 정렬 오류를 발견하고 수정하여 모델 성능을 안정화.

## 2.2 EDA 및 Preprocessing
![EDA 및 Preprocessing](https://raw.githubusercontent.com/kkogggokk/TextClassification/refs/heads/main/images/Screenshot_2024-11-25_at_1.40.53_PM.png)

- **Token 빈도 분석 및 제거:** `_num_`, `n` 등 빈도가 높은 불필요 토큰 제거, 성능 영향 미미.
- **중복 데이터 확인 및 제거:** 중복 데이터 14% 제거 후 accuracy 0.979 → 0.983으로 성능 향상.
- **문장 길이 분석:** 문장 길이에 따라 Token 최대 길이를 설정했으나, 성능 감소 확인.
- **Class Imbalance 확인:** Validation 및 Test set의 클래스 분포가 유사할 때 더 나은 성능 확인.

## 2.3 모델 선정 및 분석
![모델 선정 및 분석](https://raw.githubusercontent.com/kkogggokk/TextClassification/refs/heads/main/images/Screenshot_2024-11-25_at_1.36.47_PM.png)

- **Baseline 모델**: BERT.
- **추가 모델 실험**:
    - **RoBERTa**: BERT 대비 더 많은 데이터와 Dynamic Masking 기법 활용.
    - **ALBERT**: 파라미터 공유 및 감소로 경량화된 모델.
    - **SCI-BERT**: 학술 데이터 기반 사전 학습된 모델.
- **모델 성능**: SCI-BERT > RoBERTa > BERT > ALBERT 순서로 성능 우수.

## 2.4 모델 평가 및 개선
![모델 평가 및 개선](https://raw.githubusercontent.com/kkogggokk/TextClassification/refs/heads/main/images/Screenshot_2024-11-25_at_1.41.17_PM.png)

- **Learning Rate Scheduler 도입**: Constant Schedule, Cosine Schedule을 통해 성능 향상.
- **Ensemble (Hard Voting)**:
    - 성능이 가장 높은 3개 모델의 결과를 앙상블, 0.990 도달.
    - 추가로 5개 모델 앙상블하여 최종 0.992 성능 도달.

# 3. 프로젝트 결과

- 정확도 0.992 달성
- 교육과정 캐글 대회에서 8팀 중 1위 차지

# 4. 자체 평가 및 보완

### Pre-processing

- **한계**: 기본적인 중복 제거에 그침.
- **개선 방향**: Special Token 추가 및 위치 조정 시도.

### Model

- **한계**: BERT 류 모델만 사용.
- **개선 방향**: XLNet, CNN 등 BERT 이외 모델 활용.

### Optimizer

- **한계**: AdamW만 사용.
- **개선 방향**: RAdam, BertAdam 등 다양한 Optimizer 실험.

### Hyperparameter Tuning

- **한계**: 제공된 기본 조합만 사용.
- **개선 방향**: 다양한 조합 시도 및 성능 비교.
