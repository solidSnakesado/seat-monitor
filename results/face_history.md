# 얼굴 판정 성능 히스토리

`python -m evaluation.face_compare`를 실행할 때마다 한 줄씩 추가된다 (지우지 않고 누적).
A MediaPipe만 / B 수치만 학습 / C 이미지만 학습 / D 수치 + 이미지 학습 / E B·C 결과 결합 / T 시계열 / F T·C 결과 결합.
점수는 균형 정확도, 오경보는 앞 보기 구간(화면·강의 공간·헷갈리는 동작)을 영역 밖·졸음·하품으로 판정한 비율.

| 날짜 | 단계 | 학습 세션 | 평가 세션 | A | B | C | D | E | 오경보 A / B / C / D / E | 상세 | T | F | 오경보 T / F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-16 16:46 | 확인용 1세션 | room1_day_g1 | 검증: room1_day_g0 | 0.447 | 0.526 | 0.427 | 0.438 | 0.476 | 35% / 14% / 85% / 79% / 55% | compare_face_20261016_164649.md |
| 2026-10-16 18:45 | 메인 v1 기본 설정 | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.743 | 0.792 | 0.825 | 32% / 28% / 13% / 13% / 15% | compare_face_20261016_184540.md |
| 2026-10-16 18:52 | v1 E 비중 0.4 (곡선 최고) | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.743 | 0.792 | 0.854 | 32% / 28% / 13% / 13% / 21% | compare_face_20261016_185249.md |
| 2026-10-16 19:10 | C 스무딩+가리기 seed 2개, E 0.6 | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.842 | 0.792 | 0.871 | 32% / 28% / 17% / 13% / 22% | compare_face_20261016_191055.md |
| 2026-10-16 19:17 | D 스무딩+가리기 | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.842 | 0.801 | 0.871 | 32% / 28% / 17% / 39% / 22% | compare_face_20261016_191718.md |
| 2026-10-16 19:33 | E 영역 연결 w 0.7 (오경보 최소 기준) | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.842 | 0.792 | 0.845 | 32% / 28% / 17% / 13% / 5% | compare_face_20261016_193342.md |
| 2026-10-19 10:55 | T 시계열 conv 추가 | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.842 | 0.792 | 0.845 | 32% / 28% / 17% / 13% / 5% | compare_face_20261019_105538.md | 0.880 | 0.878 | 14% / 3% |
| 2026-10-19 11:23 | F 영역 연결 0° | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.842 | 0.792 | 0.847 | 32% / 28% / 17% / 13% / 7% | compare_face_20261019_112301.md | 0.880 | 0.880 | 14% / 5% |
| 2026-10-19 11:23 | F 영역 연결 2° | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.842 | 0.792 | 0.846 | 32% / 28% / 17% / 13% / 6% | compare_face_20261019_112309.md | 0.880 | 0.880 | 14% / 4% |
| 2026-10-19 11:30 | T conv seed 3개, F 영역 연결 5° | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.842 | 0.792 | 0.845 | 32% / 28% / 17% / 13% / 5% | compare_face_20261019_113017.md | 0.893 | 0.885 | 17% / 3% |
| 2026-10-19 14:16 | 이벤트 지표 추가, F 기준 | room1_day_g1, room1_day_g0, room2_day_g1, room2_day_g0, room1_cloth2_far_g1 | 검증: val_room1_seat2_g1 | 0.526 | 0.830 | 0.842 | 0.792 | 0.845 | 32% / 28% / 17% / 13% / 5% | compare_face_20261019_141636.md | 0.893 | 0.885 | 17% / 3% |
