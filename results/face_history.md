# 얼굴 판정 성능 히스토리

`python -m evaluation.face_compare`를 실행할 때마다 한 줄씩 추가된다 (지우지 않고 누적).
A MediaPipe만 / B 수치만 학습 / C 이미지만 학습 / D 수치 + 이미지 학습 / E B·C 결과 결합.
점수는 균형 정확도, 오경보는 앞 보기 구간(화면·강의 공간·헷갈리는 동작)을 영역 밖·졸음·하품으로 판정한 비율.

| 날짜 | 단계 | 학습 세션 | 평가 세션 | A | B | C | D | E | 오경보 A / B / C / D / E | 상세 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-16 16:46 | 확인용 1세션 | room1_day_g1 | 검증: room1_day_g0 | 0.447 | 0.526 | 0.427 | 0.438 | 0.476 | 35% / 14% / 85% / 79% / 55% | compare_face_20261016_164649.md |
| 2026-10-16 18:45 | 메인 v1 기본 설정 | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.743 | 0.792 | 0.825 | 32% / 28% / 13% / 13% / 15% | compare_face_20261016_184540.md |
| 2026-10-16 18:52 | v1 E 비중 0.4 (곡선 최고) | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.743 | 0.792 | 0.854 | 32% / 28% / 13% / 13% / 21% | compare_face_20261016_185249.md |
| 2026-10-16 19:10 | C 스무딩+가리기 seed 2개, E 0.6 | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.842 | 0.792 | 0.871 | 32% / 28% / 17% / 13% / 22% | compare_face_20261016_191055.md |
| 2026-10-16 19:17 | D 스무딩+가리기 | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.842 | 0.801 | 0.871 | 32% / 28% / 17% / 39% / 22% | compare_face_20261016_191718.md |
