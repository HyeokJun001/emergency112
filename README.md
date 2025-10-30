# 🚑 응급 이송 병원 추천 시스템

실시간 위치 추적 기반 응급 병원 추천 및 음성 증상 입력 시스템

## 주요 기능

- 🗺️ **실시간 위치 추적** (30초 자동 갱신)
- 🏥 **증상 맞춤 병원 Top 3 추천**
- 🎤 **음성으로 증상 설명** (OpenAI Whisper STT)
- 📍 **실시간 병상/장비 정보**
- 🗺️ **지도 시각화** (병원 위치 및 경로)
- 📞 **원클릭 전화 연결**

## 배포 방법

### Streamlit Cloud 배포

1. GitHub 저장소 생성 및 파일 업로드
   - `app2.py`
   - `requirements.txt`
   - `README.md`

2. [Streamlit Cloud](https://streamlit.io/cloud) 접속

3. "New app" 클릭

4. GitHub 저장소 선택

5. Main file: `app2.py` 선택

6. Deploy 클릭!

## 필요한 API 키

현재 코드에 하드코딩되어 있지만, 실제 배포 시에는 Streamlit Secrets 사용 권장:

- DATA_GO_KR_SERVICE_KEY: 공공데이터포털 응급의료 API
- KAKAO_REST_API_KEY: 카카오 지도 API
- OPENAI_API_KEY: OpenAI Whisper API

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app2.py
```

## 주의사항

- GPS 기능은 HTTPS 환경에서만 작동 (Streamlit Cloud는 자동 HTTPS)
- 모바일 브라우저에서 위치 권한 허용 필요
- 실시간 추적은 배터리 소모가 있을 수 있음

