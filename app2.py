
# er_triage_streamlit_app_v2.py
# -*- coding: utf-8 -*-
"""
[개요]
- 사용자가 말한 '현재 내 위치 잘 잡히는' 기존 er_triage_streamlit_app.py를 기반으로
  디자인/구조를 '알잘딱'하게 강화한 버전.
- UI 요소(카드/배지/컬러), 증상-설비 매칭 가중치 기반 '적합도 점수', 지도에 경로선(LineLayer) 추가,
  병원 카드에 '전화걸기/지도길찾기' 액션버튼, SBAR 자동 요약(초안)까지 제공.

[필요 키]
- 환경변수 또는 Streamlit Secrets:
  - DATA_GO_KR_SERVICE_KEY : data.go.kr (국립중앙의료원 응급의료 API) 일반키 권장
  - KAKAO_REST_API_KEY     : 카카오 Local REST 키 (좌표→행정구역/주소)
"""

import os, math, time, datetime
from typing import Dict, Any, Optional, Tuple, List
from urllib.parse import urlparse, quote_plus

import requests
import pandas as pd
import streamlit as st
import pydeck as pdk
from geopy.distance import geodesic

# ─────────────────────────────────────────────────────────
# 🔐 환경변수 로드
# ─────────────────────────────────────────────────────────
# API 키는 환경변수 또는 Streamlit Secrets에서 가져오기
# .env 파일 또는 Streamlit Cloud Secrets 사용 필수!
try:
    DATA_GO_KR_KEY = st.secrets.get("DATA_GO_KR_SERVICE_KEY", os.getenv("DATA_GO_KR_SERVICE_KEY", ""))
    KAKAO_KEY      = st.secrets.get("KAKAO_REST_API_KEY", os.getenv("KAKAO_REST_API_KEY", ""))
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
except:
    DATA_GO_KR_KEY = os.getenv("DATA_GO_KR_SERVICE_KEY", "")
    KAKAO_KEY      = os.getenv("KAKAO_REST_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# OpenAI 클라이언트 초기화
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    openai_client = None

# ─────────────────────────────────────────────────────────
# 🌐 엔드포인트 (https 강제)
# ─────────────────────────────────────────────────────────
ER_BED_URL       = "https://apis.data.go.kr/B552657/ErmctInfoInqireService/getEmrrmRltmUsefulSckbdInfoInqire"
EGET_BASE_URL    = "https://apis.data.go.kr/B552657/ErmctInfoInqireService/getEgytBassInfoInqire"
KAKAO_C2REG_URL  = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
KAKAO_C2ADDR_URL = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
KAKAO_ADDR_URL   = "https://dapi.kakao.com/v2/local/search/address.json"

# ─────────────────────────────────────────────────────────
# 🎨 간단 CSS (카드/배지/헤더)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 페이지 폭 늘리기 */
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
/* 헤더 그라디언트 */
.header-box {
  background: linear-gradient(90deg, #e0f2fe 0%, #bae6fd 100%);
  border-radius: 14px; padding: 16px 18px; color: #0c4a6e; margin-bottom: 12px;
  border: 2px solid #38bdf8;
}
.header-box h2 {
  color: #0c4a6e !important;
  font-weight: 800 !important;
}
.header-box .small {
  color: #0369a1 !important;
}
.metric-chip {
  display: inline-block; background: #f0f4ff; color: #1b3a8a; border: 1px solid #dbe4ff;
  padding: 6px 10px; border-radius: 999px; font-size: 12px; margin-right: 8px;
}
.card {
  border: 1px solid #e6e8ec; border-radius: 12px; padding: 14px; margin-bottom: 10px;
  background: #fff;
}
.card h4 { margin: 0 0 8px 0; }
.kv { font-size: 13px; color: #333; }
.kv b { color: #111; }
.small { font-size: 12px; color: #555; }
.badge {
  display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 11px; margin-right: 6px;
}
.badge.y { background:#e7f9ed; color:#0a6b2d; border:1px solid #b9e7c9; }
.badge.n { background:#fff1f0; color:#b00020; border:1px solid #ffd6d2; }
.badge.na { background:#f5f7fa; color:#4a5568; border:1px solid #e2e8f0; }
.score {
  display:inline-block; font-weight:700; padding:4px 10px; border-radius:999px; background:#f0f7ff; color:#0b61a4; border:1px solid #cfe3ff;
}
.action-row a {
  text-decoration:none; display:inline-block; padding:6px 10px; border-radius:8px; border:1px solid #e5e7eb;
  margin-right:8px; font-size:12px; background:#fafafa;
}
.action-row a:hover { background:#f1f5f9; }
hr.hr { border:none; border-top:1px dashed #e5e7eb; margin:12px 0; }

/* 🎤 녹음 카드 스타일 */
.voice-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 16px;
  padding: 24px;
  margin: 16px 0;
  color: white;
  text-align: center;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}
.voice-card h3 {
  margin: 0 0 12px 0;
  font-size: 24px;
  font-weight: 700;
}
.voice-card p {
  margin: 0 0 16px 0;
  font-size: 14px;
  opacity: 0.9;
}
/* 오디오 입력 위젯 스타일 조정 */
.stAudioInput {
  margin: 0 auto;
}
.stAudioInput > label {
  font-size: 16px !important;
  font-weight: 600 !important;
  color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# 🌐 공통 GET (data.go.kr에만 serviceKey 부착)
# ─────────────────────────────────────────────────────────
def _http_get(url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> requests.Response:
    """
    - data.go.kr 요청에만 serviceKey를 붙여 403 방지
    - http→https 승격
    - 타임아웃 및 status 체크
    """
    timeout = (5, 15)
    params = dict(params) if params else {}
    if url.startswith("http://apis.data.go.kr/"):
        url = url.replace("http://", "https://", 1)
    netloc = urlparse(url).netloc
    if "apis.data.go.kr" in netloc:
        svc_key = params.pop("serviceKey", DATA_GO_KR_KEY)
        if svc_key:
            if "%" in svc_key:  # 인코딩키일 경우 쿼리에 직접 삽입
                join = "&" if ("?" in url) else "?"
                url = f"{url}{join}serviceKey={svc_key}"
            else:
                params["serviceKey"] = svc_key
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp

# ─────────────────────────────────────────────────────────
# 🗺️ Kakao Local 유틸 (좌표↔행정구역/주소)
# ─────────────────────────────────────────────────────────
def kakao_coord2region(lon: float, lat: float, kakao_key: str) -> Optional[Tuple[str, str, str]]:
    """ (경도, 위도) → (시/도, 시/군/구, code)  (법정동 우선) """
    if not kakao_key: return None
    headers = {"Authorization": f"KakaoAK {kakao_key}"}
    params = {"x": lon, "y": lat}
    try:
        r = _http_get(KAKAO_C2REG_URL, params=params, headers=headers)
        docs = r.json().get("documents", [])
        target = next((d for d in docs if d.get("region_type") == "B"), docs[0] if docs else None)
        if not target: return None
        return target.get("region_1depth_name"), target.get("region_2depth_name"), target.get("code")
    except Exception:
        return None

def kakao_coord2address(lon: float, lat: float, kakao_key: str) -> Optional[str]:
    """ (경도, 위도) → 주소 문자열(도로명 우선/지번 보조) """
    if not kakao_key: return None
    headers = {"Authorization": f"KakaoAK {kakao_key}"}
    params = {"x": lon, "y": lat}
    try:
        r = _http_get(KAKAO_C2ADDR_URL, params=params, headers=headers)
        docs = r.json().get("documents", [])
        if not docs: return None
        d0 = docs[0]
        if d0.get("road_address") and d0["road_address"].get("address_name"):
            return d0["road_address"]["address_name"]
        if d0.get("address") and d0["address"].get("address_name"):
            return d0["address"]["address_name"]
        return None
    except Exception:
        return None

# ─────────────────────────────────────────────────────────
# 🏥 병원 기본/병상 조회
# ─────────────────────────────────────────────────────────
def fetch_baseinfo_by_hpid(hpid: str, service_key: str) -> Optional[Dict[str, Any]]:
    """ HPID → {좌표, 주소, 응급전화, 병원명} """
    from xml.etree import ElementTree as ET
    try:
        r = _http_get(EGET_BASE_URL, {"HPID": hpid, "pageNo": 1, "numOfRows": 1, "serviceKey": service_key})
        it = ET.fromstring(r.content).find(".//item")
        if it is None: return None
        def g(tag):
            el = it.find(tag)
            return el.text.strip() if el is not None and el.text is not None else None
        return {
            "hpid": g("hpid"),
            "dutyName": g("dutyName") or g("dutyname"),
            "dutyAddr": g("dutyAddr"),
            "dutytel3": g("dutyTel3"),
            "wgs84Lat": float(g("wgs84Lat")) if g("wgs84Lat") else None,
            "wgs84Lon": float(g("wgs84Lon")) if g("wgs84Lon") else None,
        }
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_emergency_hospitals_in_region(sido: str, sigungu: str, service_key: str, max_items: int = 250) -> pd.DataFrame:
    """
    1) 실시간 병상 API에서 지역별 HPID 수집
    2) HPID별 기본정보 조회(좌표/주소/전화)
    """
    from xml.etree import ElementTree as ET
    r = _http_get(ER_BED_URL, {"STAGE1": sido, "STAGE2": sigungu, "pageNo": 1, "numOfRows": 1000, "serviceKey": service_key})
    root = ET.fromstring(r.content)

    hpids = []
    for it in root.findall(".//item"):
        el = it.find("hpid")
        if el is not None and el.text:
            hpids.append(el.text.strip())
    hpids = list(dict.fromkeys(hpids))[:max_items]

    rows = []
    for h in hpids:
        info = fetch_baseinfo_by_hpid(h, service_key)
        if info and info.get("wgs84Lat") and info.get("wgs84Lon"):
            rows.append(info)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["wgs84Lat", "wgs84Lon"])
    return df

@st.cache_data(ttl=30)
def fetch_er_beds(sido: str, sigungu: str, service_key: str, rows: int = 1000) -> pd.DataFrame:
    """ 실시간 응급 병상/장비 지표 DF """
    from xml.etree import ElementTree as ET
    params = {"STAGE1": sido, "STAGE2": sigungu, "pageNo": 1, "numOfRows": rows, "serviceKey": service_key}
    resp = _http_get(ER_BED_URL, params=params)
    items = ET.fromstring(resp.content).findall(".//item")

    rows_list = []
    for it in items:
        def g(tag):
            el = it.find(tag)
            return el.text.strip() if el is not None and el.text is not None else None
        rows_list.append({
            "hpid": g("hpid"),
            "dutyName": g("dutyname"),
            "hvidate": g("hvidate"),
            # 핵심 병상/장비
            "hvec": g("hvec"), "hvoc": g("hvoc"), "hvicc": g("hvicc"), "hvgc": g("hvgc"),
            "hvcc": g("hvcc"), "hvncc": g("hvncc"), "hvccc": g("hvccc"),
            "hvctayn": g("hvctayn"), "hvmriayn": g("hvmriayn"), "hvangioayn": g("hvangioayn"), "hvventiayn": g("hvventiayn"),
            # 세부 병상 코드
            "hv1": g("hv1"), "hv2": g("hv2"), "hv3": g("hv3"), "hv4": g("hv4"),
            "hv5": g("hv5"), "hv6": g("hv6"), "hv7": g("hv7"), "hv8": g("hv8"),
            "hv9": g("hv9"), "hv10": g("hv10"), "hv11": g("hv11"), "hv12": g("hv12"),
            "dutytel3": g("dutytel3") or g("hv1"),
            "hvdnm": g("hvdnm"),
        })
    df = pd.DataFrame(rows_list)
    if not df.empty:
        df = df.dropna(subset=["hpid"])
    return df

# ─────────────────────────────────────────────────────────
# 📏 거리 + 간이 적합도 점수
# ─────────────────────────────────────────────────────────
def add_distance_km(df: pd.DataFrame, user_lat: float, user_lon: float) -> pd.DataFrame:
    if df.empty:
        df["distance_km"] = []
        return df
    out = df.copy()
    dists = []
    for _, row in out.iterrows():
        try:
            d = geodesic((user_lat, user_lon), (float(row["wgs84Lat"]), float(row["wgs84Lon"]))).km
        except Exception:
            d = math.nan
        dists.append(d)
    out["distance_km"] = dists
    return out

def _safe_int(x):
    try:
        s = str(x).strip()
        return int(s) if s not in ("", "None", "nan") else 0
    except:
        return 0

# 증상 규칙 + 가중치
SYMPTOM_RULES = {
    "뇌졸중 의심(FAST+)": {
        "must":  [("hvctayn","Y"), ("hvicc",1)],
        "bonus": [("hv5",1),("hv6",1)],
        "weights": {"distance": -2.0, "beds_core": 1.2, "equip": 4.0},
        "explain": "CT 가능(Y) + 중환자실 가용(일반중환자 hvicc≥1)이 핵심",
    },
    "심근경색 의심(STEMI)": {
        "must":  [("hvangioayn","Y"), ("hvoc",1), ("hvicc",1)],
        "bonus": [],
        "weights": {"distance": -2.0, "beds_core": 1.1, "equip": 4.5},
        "explain": "혈관조영(Y)+수술실/ICU 가용",
    },
    "다발성 외상/중증 외상": {
        "must":  [("hvventiayn","Y"), ("hvoc",1), ("hvicc",1)],
        "bonus": [("hv9",1)],
        "weights": {"distance": -2.2, "beds_core": 1.3, "equip": 3.8},
        "explain": "인공호흡(Y)+수술실/ICU 가용",
    },
    "소아 중증(호흡곤란/경련 등)": {
        "must":  [("hvncc",1)],  # 신생중환자실
        "bonus": [("hv10","Y"),("hv11","Y")],  # 소아 VENTI/인큐
        "weights": {"distance": -1.8, "beds_core": 1.2, "equip": 3.5},
        "explain": "신생중환자실 가용이 핵심",
    },
    "정형외과 중증(대형골절/절단)": {
        "must":  [("hvoc",1), ("hv3",1), ("hvicc",1)],
        "bonus": [("hv4",1)],
        "weights": {"distance": -2.1, "beds_core": 1.0, "equip": 3.0},
        "explain": "수술실+외과계 중환자/입원실",
    },
    "신경외과 응급(의식저하/외상성출혈)": {
        "must":  [("hvctayn","Y"), ("hv6",1), ("hvicc",1)],
        "bonus": [],
        "weights": {"distance": -2.0, "beds_core": 1.1, "equip": 4.0},
        "explain": "CT 가능 + 신경외과 중환자실",
    },
}

def check_must(row: pd.Series, must: List[Tuple[str, Any]]) -> bool:
    """필수 조건 충족 여부"""
    for key, want in must:
        val = row.get(key)
        if isinstance(want, str):
            if str(val).strip().upper() != want:
                return False
        else:
            if _safe_int(val) < int(want):
                return False
    return True

def suitability_score(row: pd.Series, symptom: str) -> float:
    """
    [간이 적합도 점수] 0~100 스케일 (해커톤 데모용 휴리스틱)
    - distance: km당 패널티 (멀수록 감점)
    - beds_core: 핵심 병상(응급실/수술실/ICU/입원실 등) 합산 가점
    - equip: 필수/보너스 설비(CT/ANGIO/VENTI/신생아 등) 가점
    """
    rule = SYMPTOM_RULES[symptom]
    w = rule["weights"]

    # 거리 (작을수록 좋음) → 음수 가중치
    d = float(row.get("distance_km") or 0.0)
    score = w["distance"] * d  # (-)감점

    # 가용 병상 핵심 합산 (없으면 0)
    beds = _safe_int(row.get("hvec")) + _safe_int(row.get("hvoc")) + _safe_int(row.get("hvicc")) + _safe_int(row.get("hvgc"))
    score += w["beds_core"] * min(beds, 50)  # 상한 50

    # 장비/전문성 플래그 (Y/정수) → 보너스
    equip_flags = 0
    for key, want in rule.get("bonus", []):
        val = row.get(key)
        equip_flags += 1 if (str(val).strip().upper() == "Y" or _safe_int(val) >= int(want)) else 0

    # 공통적으로 중요한 장비(Y)
    for key in ["hvctayn","hvangioayn","hvventiayn","hvmriayn"]:
        val = row.get(key)
        equip_flags += 1 if str(val).strip().upper() == "Y" else 0

    score += w["equip"] * equip_flags

    # 0~100로 클리핑
    score = max(0.0, min(100.0, 50.0 + score/2.0))  # 기준점 50, 스케일 조정
    return round(score, 1)

# ─────────────────────────────────────────────────────────
# 🌐 GPS + IP Fallback (자동)
# ─────────────────────────────────────────────────────────
def get_geolocation_auto() -> Tuple[Optional[float], Optional[float], str]:
    """
    1) streamlit_js_eval.get_geolocation() 우선 시도
    2) 실패 시 IP 기반 좌표 추정
    - 반환: (lat, lon, source)  source ∈ {"GPS","IP",""}
    """
    # 1) 브라우저 GPS (HTTPS/localhost 필요)
    try:
        from streamlit_js_eval import get_geolocation
        loc = get_geolocation()
        if loc and isinstance(loc, dict):
            coords = loc.get("coords") or loc
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            if lat is not None and lon is not None:
                return float(lat), float(lon), "GPS"
    except Exception:
        pass

    # 2) 실패 시 IP
    try:
        r = requests.get("https://ipapi.co/json/", timeout=4)
        if r.ok:
            j = r.json()
            lat, lon = j.get("latitude"), j.get("longitude")
            if lat and lon:
                return float(lat), float(lon), "IP"
    except Exception:
        pass

    try:
        r = requests.get("https://ipinfo.io/json", timeout=4)
        if r.ok:
            j = r.json()
            if "loc" in j:
                lat_str, lon_str = j["loc"].split(",")
                return float(lat_str), float(lon_str), "IP"
    except Exception:
        pass

    return None, None, ""

# ─────────────────────────────────────────────────────────
# 🎤 OpenAI Whisper STT (Speech-to-Text)
# ─────────────────────────────────────────────────────────
def transcribe_audio(audio_bytes) -> Optional[str]:
    """
    음성 파일(bytes)을 OpenAI Whisper API로 텍스트 변환
    - 반환: 텍스트 문자열 또는 None
    """
    if not OPENAI_AVAILABLE or not openai_client:
        return None
    
    try:
        # 임시 파일로 저장 (Whisper API는 file-like object 필요)
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Whisper API 호출
        with open(tmp_file_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko"  # 한국어 우선
            )
        
        # 임시 파일 삭제
        os.unlink(tmp_file_path)
        
        return transcript.text.strip() if transcript.text else None
    
    except Exception as e:
        st.error(f"음성 변환 오류: {e}")
        return None

# ─────────────────────────────────────────────────────────
# 🖥️ Streamlit UI
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="증상맞춤 응급 병상 Top3 (Polished)", page_icon="🚑", layout="wide")
st.markdown('<div class="header-box"><h2 style="margin:0;">🚑 증상 맞춤 응급 이송 보조</h2><div class="small">현재 위치 기반 · 실시간 병상/장비 · 증상 적합도 Top3</div></div>', unsafe_allow_html=True)

# 세션 상태
for k, v in [("lat", None), ("lon", None), ("addr", ""), ("sido", None), ("sigungu", None), ("loc_source",""), ("voice_text", "")]:
    if k not in st.session_state: st.session_state[k] = v

# 🚨 실시간 자동 위치 추적 모드
if 'auto_tracking' not in st.session_state:
    st.session_state.auto_tracking = False
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None

# 상단 컨트롤 바
bar1, bar2, bar3, bar4 = st.columns([1.6,1,1,1])
with bar1:
    # 실시간 추적 토글
    tracking_enabled = st.toggle(
        "🚑 실시간 위치 추적 (30초마다 자동 갱신)",
        value=st.session_state.auto_tracking,
        help="엠뷸런스 이동 중 자동으로 위치를 계속 갱신합니다"
    )
    
    if tracking_enabled != st.session_state.auto_tracking:
        st.session_state.auto_tracking = tracking_enabled
        if tracking_enabled:
            st.success("✅ 실시간 위치 추적 활성화!")
        else:
            st.info("⏸️ 실시간 위치 추적 일시정지")
        st.rerun()
    
    # 수동 갱신 버튼
    if st.button("📍 지금 즉시 위치 갱신", type="primary", use_container_width=True):
        with st.spinner("📍 위치 정보를 가져오는 중..."):
            lat, lon, src = get_geolocation_auto()
            
            if lat is not None and lon is not None:
                st.session_state["lat"] = lat
                st.session_state["lon"] = lon
                st.session_state["loc_source"] = src
                st.session_state.last_update_time = time.time()
                
                # 주소/행정구역 파생
                if KAKAO_KEY:
                    addr = kakao_coord2address(lon, lat, KAKAO_KEY) or ""
                    st.session_state["addr"] = addr
                    reg = kakao_coord2region(lon, lat, KAKAO_KEY)
                    if reg:
                        st.session_state["sido"], st.session_state["sigungu"], _ = reg
                
                st.success(f"✅ 위치 갱신 완료! ({src}) - 위도: {lat:.4f}, 경도: {lon:.4f}")
                st.rerun()
            else:
                st.error(f"❌ 위치를 가져올 수 없습니다. 수동으로 입력하세요.")
                
                # 수동 입력 옵션
                col_lat, col_lon = st.columns(2)
                with col_lat:
                    manual_lat = st.number_input("위도", value=35.6, format="%.6f")
                with col_lon:
                    manual_lon = st.number_input("경도", value=126.5, format="%.6f")
                
                if st.button("✅ 수동 위치 적용"):
                    st.session_state["lat"] = manual_lat
                    st.session_state["lon"] = manual_lon
                    st.session_state["loc_source"] = "Manual"
                    st.session_state.last_update_time = time.time()
                    
                    if KAKAO_KEY:
                        addr = kakao_coord2address(manual_lon, manual_lat, KAKAO_KEY) or ""
                        st.session_state["addr"] = addr
                        reg = kakao_coord2region(manual_lon, manual_lat, KAKAO_KEY)
                        if reg:
                            st.session_state["sido"], st.session_state["sigungu"], _ = reg
                    
                    st.success("✅ 수동 위치 설정 완료!")
                    st.rerun()

# 🔄 자동 위치 갱신 (30초마다) - 백그라운드에서 조용히
if st.session_state.auto_tracking:
    current_time = time.time()
    last_update = st.session_state.last_update_time or 0
    time_until_next = 30 - (current_time - last_update)
    
    # 30초 경과 체크 - 위치만 조용히 갱신
    if time_until_next <= 0 or last_update == 0:
        # 백그라운드에서 조용히 위치 가져오기 (화면 새로고침 없이)
        lat, lon, src = get_geolocation_auto()
        
        if lat is not None and lon is not None:
            # 위치가 변경되었는지 확인
            location_changed = (
                st.session_state["lat"] != lat or 
                st.session_state["lon"] != lon
            )
            
            st.session_state["lat"] = lat
            st.session_state["lon"] = lon
            st.session_state["loc_source"] = src
            st.session_state.last_update_time = time.time()
            
            # 주소/행정구역 파생
            if KAKAO_KEY:
                addr = kakao_coord2address(lon, lat, KAKAO_KEY) or ""
                st.session_state["addr"] = addr
                reg = kakao_coord2region(lon, lat, KAKAO_KEY)
                if reg:
                    st.session_state["sido"], st.session_state["sigungu"], _ = reg
            
            # 위치가 실제로 변경된 경우에만 조용히 알림 (새로고침 없이)
            if location_changed:
                st.toast(f"📍 위치 업데이트: {addr[:20]}...", icon="🔄")
        else:
            st.session_state.last_update_time = time.time()
    
    # 다음 갱신까지 남은 시간 표시 (작게)
    if time_until_next > 0:
        st.caption(f"⏱️ 자동 추적 활성화 중... 다음 갱신: {int(time_until_next)}초 후")

with bar2:
    # 증상 선택
    symptom = st.selectbox("환자 증상", list(SYMPTOM_RULES.keys()), index=0)
with bar3:
    # 정렬 우선순위
    sort_pref = st.selectbox("정렬 기준", ["적합도 점수", "가까운 순"], index=0)
with bar4:
    auto_refresh = st.toggle("자동 갱신", value=True, help="위치/증상 변경 시 자동 조회")

# 현재 좌표/행정구역 표시
loc_col1, loc_col2, loc_col3 = st.columns([2,1,1])
with loc_col1:
    st.markdown(f'<span class="metric-chip">좌표: {st.session_state["lat"] or "—"}, {st.session_state["lon"] or "—"}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="metric-chip">주소: {st.session_state["addr"] or "—"}</span>', unsafe_allow_html=True)
with loc_col2:
    src = st.session_state.get("loc_source") or "—"
    src_label = "GPS" if src=="GPS" else ("IP 추정" if src=="IP" else "미확정")
    st.markdown(f'<span class="metric-chip">위치소스: {src_label}</span>', unsafe_allow_html=True)
with loc_col3:
    rg = f'{st.session_state["sido"] or "—"} {st.session_state["sigungu"] or ""}'.strip()
    st.markdown(f'<span class="metric-chip">행정구역: {rg}</span>', unsafe_allow_html=True)

st.divider()

# 🎤 음성 입력 섹션 (지도 아래로 이동)
if OPENAI_AVAILABLE:
    # 녹음 모드 상태
    if 'voice_recording_mode' not in st.session_state:
        st.session_state.voice_recording_mode = False
    
    # 녹음 모드가 아닐 때: 큰 보라색 카드 버튼
    if not st.session_state.voice_recording_mode:
        # 큰 보라색 카드를 버튼으로 구현
        card_clicked = st.button(
            "🎤\n\n음성으로 증상 설명하기\n\n이 카드를 눌러 녹음을 시작하세요",
            key="voice_card_button",
            use_container_width=True,
            type="primary"
        )
        
        # CSS로 버튼을 크고 보라색으로 스타일링
        st.markdown("""
        <style>
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            padding: 60px 40px !important;
            font-size: 28px !important;
            font-weight: 700 !important;
            line-height: 1.8 !important;
            white-space: pre-line !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
            transition: transform 0.2s, box-shadow 0.2s !important;
            min-height: 250px !important;
            color: white !important;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
            background: linear-gradient(135deg, #7a8ff0 0%, #8b5bb0 100%) !important;
            color: white !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:active {
            transform: translateY(0px) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if card_clicked:
            st.session_state.voice_recording_mode = True
            st.rerun()
    
    else:
        # 녹음 모드: 녹음 중 UI 표시
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    border-radius: 16px; padding: 40px; text-align: center; margin: 16px 0;
                    box-shadow: 0 4px 12px rgba(240, 147, 251, 0.4);
                    animation: pulse 1.5s infinite;">
            <div style="font-size: 80px; margin-bottom: 20px;">⏺️</div>
            <h3 style="color: white; font-size: 28px; margin: 0 0 12px 0; font-weight: 700;">녹음 중...</h3>
            <p style="color: rgba(255, 255, 255, 0.9); font-size: 16px; margin: 0;">
                증상을 자세히 설명해주세요
            </p>
        </div>
        <style>
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Streamlit 내장 오디오 입력 사용
        audio_data = st.audio_input("🎙️ 녹음하세요", key="audio_input_recording")
        
        if audio_data is not None:
            audio_bytes = audio_data.getvalue()
            
            with st.spinner("🎧 음성을 텍스트로 변환하는 중..."):
                text = transcribe_audio(audio_bytes)
                
                if text:
                    st.session_state["voice_text"] = text
                    st.session_state.voice_recording_mode = False
                    st.success("✅ 음성 변환 완료!")
                    st.rerun()
        
        # 취소 버튼
        if st.button("❌ 녹음 취소", use_container_width=True):
            st.session_state.voice_recording_mode = False
            st.rerun()
    
    # 초기화 버튼
    if st.session_state.get("voice_text"):
        if st.button("🗑️ 음성 텍스트 초기화", use_container_width=True, type="secondary"):
            st.session_state["voice_text"] = ""
            st.rerun()
    
    # 변환된 텍스트 표시
    if st.session_state.get("voice_text"):
        st.markdown("---")
        st.markdown("### 📝 변환된 텍스트")
        
        # 큰 텍스트 박스로 표시
        st.markdown(f"""
        <div style="background: #f0f9ff; border-left: 4px solid #0284c7; padding: 16px 20px; border-radius: 8px; margin: 12px 0;">
            <p style="margin: 0; font-size: 16px; line-height: 1.6; color: #0c4a6e;">
                {st.session_state["voice_text"]}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 텍스트 편집 가능
        with st.expander("✏️ 내용 수정하기"):
            edited_text = st.text_area(
                "변환된 텍스트를 수정할 수 있습니다:",
                value=st.session_state["voice_text"],
                height=120,
                key="voice_text_edit"
            )
            
            col_save, col_cancel = st.columns([1, 1])
            with col_save:
                if st.button("💾 저장", use_container_width=True, type="primary", key="save_voice"):
                    st.session_state["voice_text"] = edited_text
                    st.success("저장되었습니다!")
                    st.rerun()
            with col_cancel:
                if st.button("❌ 전체 삭제", use_container_width=True, key="delete_voice"):
                    st.session_state["voice_text"] = ""
                    st.rerun()
else:
    st.warning("⚠️ OpenAI API를 사용할 수 없습니다. openai 라이브러리를 설치하세요: `pip install openai`")

st.divider()

# 조회 실행 함수 (UI 반응형)
def run_query_and_render():
    # 필수 키 체크
    if not DATA_GO_KR_KEY:
        st.error("❌ DATA_GO_KR_SERVICE_KEY가 설정되지 않았습니다.")
        st.warning("환경 변수 또는 코드에 API 키를 설정해주세요.")
        return
    
    if st.session_state["lat"] is None or st.session_state["lon"] is None:
        st.info("먼저 상단의 '📍 내 위치 재설정'을 눌러 위치를 확보하세요.")
        return
    
    # 디버깅 정보 표시
    with st.expander("🔍 디버깅 정보 (개발용)", expanded=False):
        st.write(f"- 위도: {st.session_state['lat']}")
        st.write(f"- 경도: {st.session_state['lon']}")
        st.write(f"- 시/도: {st.session_state.get('sido')}")
        st.write(f"- 시/군/구: {st.session_state.get('sigungu')}")
        st.write(f"- DATA_GO_KR_KEY: {'설정됨' if DATA_GO_KR_KEY else '없음'}")
        st.write(f"- 자동 갱신: {auto_refresh}")

    # 행정구역 보강 로직 (카카오 실패시 시/도만 추정하여 진행)
    sido = st.session_state.get("sido")
    sigungu = st.session_state.get("sigungu")
    if not sido:
        # 주소에서 시/도 대략 추정
        addr = st.session_state.get("addr") or ""
        parts = addr.split()
        if len(parts) >= 1:
            sido = parts[0]
        else:
            sido = "광주광역시"  # 아주 보수적 기본값
        if len(parts) >= 2:
            sigungu = parts[1]

    user_lat, user_lon = float(st.session_state["lat"]), float(st.session_state["lon"])

    # 데이터 조회
    try:
        with st.spinner("🏥 병원 좌표 조회 중..."):
            hospitals = fetch_emergency_hospitals_in_region(sido, sigungu or "", DATA_GO_KR_KEY, max_items=300)
            st.write(f"✅ 병원 {len(hospitals)}곳 조회 완료")
    except Exception as e:
        st.error(f"❌ 병원 정보 조회 실패: {str(e)}")
        return

    try:
        with st.spinner("📡 실시간 병상/장비 조회 중..."):
            beds = fetch_er_beds(sido, sigungu or "", DATA_GO_KR_KEY, rows=1000)
            st.write(f"✅ 병상 정보 {len(beds)}건 조회 완료")
    except Exception as e:
        st.error(f"❌ 병상 정보 조회 실패: {str(e)}")
        return

    if hospitals.empty:
        st.error(f"❌ '{sido} {sigungu}' 지역의 병원 기본정보를 찾을 수 없습니다.")
        st.info("다른 지역을 시도하거나, 시/도명이 정확한지 확인해주세요.")
        return
    
    if beds.empty:
        st.error(f"❌ '{sido} {sigungu}' 지역의 실시간 병상 정보를 찾을 수 없습니다.")
        st.info("API 응답이 없거나 해당 지역에 데이터가 없을 수 있습니다.")
        return

    merged = pd.merge(hospitals, beds, on="hpid", how="left", suffixes=("", "_bed"))
    if merged.empty:
        st.error("병원 기본정보와 병상 정보가 조인되지 않았습니다.")
        return

    # 거리/점수 계산
    merged = add_distance_km(merged, user_lat, user_lon)
    merged["score"] = merged.apply(lambda r: suitability_score(r, symptom), axis=1)

    # 필수 조건 필터
    must_ok = merged[merged.apply(lambda r: check_must(r, SYMPTOM_RULES[symptom]["must"]), axis=1)].copy()
    if must_ok.empty:
        st.warning("⚠️ 증상 필수 조건을 만족하는 병원이 없어, 전체에서 정렬만 적용합니다.")
        df_show = merged.copy()
    else:
        df_show = must_ok.copy()

    # 정렬
    if sort_pref == "적합도 점수":
        df_show = df_show.sort_values(["score","distance_km"], ascending=[False, True])
    else:
        df_show = df_show.sort_values(["distance_km","score"], ascending=[True, False])

    # Top3
    top3 = df_show.head(3).copy()

    # 요약 메트릭
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("탐색 병원 수", len(merged))
    with m2: st.metric("조건 충족 수", len(must_ok) if not must_ok.empty else 0)
    with m3: st.metric("정렬 기준", sort_pref)
    with m4: st.metric("갱신", datetime.datetime.now().strftime("%H:%M:%S"))

    # 지도 (사용자 → 각 병원 라인) - Top3 앞으로 이동
    st.markdown("#### 🗺️ 지도")
    user_latlon = [{"lat": float(user_lat), "lon": float(user_lon)}]
    user_layer = pdk.Layer(
        "ScatterplotLayer", data=user_latlon, get_position="[lon, lat]", get_radius=70, pickable=False
    )
    hosp_markers = []
    line_data = []
    for _, r in top3.iterrows():
        lat, lon = float(r["wgs84Lat"]), float(r["wgs84Lon"])
        hosp_markers.append({"lat": lat, "lon": lon, "name": r.get("dutyName"), "addr": r.get("dutyAddr")})
        line_data.append({"from_lon": user_lon, "from_lat": user_lat, "to_lon": lon, "to_lat": lat})
    hosp_layer = pdk.Layer("ScatterplotLayer", data=hosp_markers, get_position="[lon, lat]", get_radius=80, pickable=True)
    line_layer = pdk.Layer(
        "LineLayer", data=line_data, get_source_position="[from_lon, from_lat]", get_target_position="[to_lon, to_lat]",
        get_width=3, pickable=False
    )
    tooltip = {"html": "<b>{name}</b><br/>{addr}", "style": {"backgroundColor": "steelblue", "color": "white"}}
    mid_lat, mid_lon = user_lat, user_lon
    if hosp_markers:
        avg_lat = sum(m["lat"] for m in hosp_markers)/len(hosp_markers)
        avg_lon = sum(m["lon"] for m in hosp_markers)/len(hosp_markers)
        mid_lat = (user_lat + avg_lat)/2; mid_lon = (user_lon + avg_lon)/2
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=12),
        layers=[user_layer, hosp_layer, line_layer],
        tooltip=tooltip,
    ))

    st.markdown("---")
    st.markdown("#### 🏆 추천 병원 Top 3")
    # 카드 렌더
    for _, r in top3.iterrows():
        # 장비 배지
        def yn_badge(key, label):
            v = str(r.get(key) or "").upper()
            cls = "y" if v=="Y" else ("na" if v=="" else "n")
            return f'<span class="badge {cls}">{label}:{v if v else "N/A"}</span>'

        tel = (r.get("dutytel3") or "").strip()
        addr = (r.get("dutyAddr") or "").strip()
        name = (r.get("dutyName") or "").strip()
        lat, lon = float(r["wgs84Lat"]), float(r["wgs84Lon"])
        dist = r.get("distance_km")
        score = r.get("score")

        kakao_nav = f"https://map.kakao.com/link/to/{quote_plus(name)},{lat},{lon}"
        gmaps_nav = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        tel_link  = f"tel:{tel}" if tel else ""

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<h4>{name} <span class='score'>{score} 점</span></h4>", unsafe_allow_html=True)
        
        # 거리 정보 (눈에 띄게)
        st.markdown(f"""
        <div style="background: #f0f9ff; padding: 12px 16px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #0284c7;">
            <div style="font-size: 16px; font-weight: 600; color: #0369a1; margin-bottom: 4px;">
                📍 거리: <span style="color: #0c4a6e;">{dist:.2f} km</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 전화번호 (큰 글씨로 강조)
        if tel:
            st.markdown(f"""
            <div style="background: #fef3c7; padding: 16px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #f59e0b;">
                <div style="font-size: 14px; color: #92400e; margin-bottom: 4px;">응급실 전화</div>
                <a href="tel:{tel}" style="font-size: 24px; font-weight: 700; color: #b45309; text-decoration: none; display: block;">
                    📞 {tel}
                </a>
                <div style="font-size: 12px; color: #92400e; margin-top: 6px;">터치하면 바로 전화 연결됩니다</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #fee2e2; padding: 12px; border-radius: 8px; margin: 10px 0;">
                <div style="font-size: 14px; color: #991b1b;">📞 전화번호 정보 없음</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 주소 (큰 글씨로 강조)
        st.markdown(f"""
        <div style="background: #e0f2fe; padding: 16px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #38bdf8;">
            <div style="font-size: 14px; color: #075985; margin-bottom: 6px;">🏥 병원 주소</div>
            <div style="font-size: 16px; font-weight: 600; color: #0c4a6e; line-height: 1.5;">
                {addr or '주소 정보 없음'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 당직의 정보
        hvdnm = r.get('hvdnm') or '—'
        if hvdnm != '—':
            st.markdown(f"""
            <div style="font-size: 14px; color: #374151; margin: 8px 0;">
                👨‍⚕️ <b>당직의:</b> {hvdnm}
            </div>
            """, unsafe_allow_html=True)
        
        # 필수조건 및 장비/병상 정보
        st.markdown('<div class="small" style="margin-top: 12px;">필수조건: ' + SYMPTOM_RULES[symptom]["explain"] + '</div>', unsafe_allow_html=True)
        st.markdown('<div class="small">장비상태: ' +
                    yn_badge("hvctayn","CT") + yn_badge("hvmriayn","MRI") + yn_badge("hvangioayn","ANGIO") + yn_badge("hvventiayn","VENTI") +
                    '</div>', unsafe_allow_html=True)
        st.markdown('<div class="small">병상(핵심): ' +
                    f'응급실 {r.get("hvec") or 0} · 수술실 {r.get("hvoc") or 0} · 일반중환자 {r.get("hvicc") or 0} · 입원실 {r.get("hvgc") or 0}' +
                    '</div>', unsafe_allow_html=True)
        
        # 길찾기 버튼 (더 크게)
        st.markdown('<hr class="hr"/>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display: flex; gap: 8px; margin-top: 12px;">
            <a href="{kakao_nav}" target="_blank" style="flex: 1; text-align: center; padding: 14px; background: #FEE500; color: #3c1e1e; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 15px;">
                🗺️ 카카오맵 길찾기
            </a>
            <a href="{gmaps_nav}" target="_blank" style="flex: 1; text-align: center; padding: 14px; background: #4285f4; color: white; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 15px;">
                🌍 구글맵 길찾기
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # SBAR 초안(복사용)
    st.markdown("#### 🧾 SBAR 자동 요약(초안)")
    
    # 음성 텍스트가 있으면 추가
    voice_detail = ""
    if st.session_state.get("voice_text"):
        voice_detail = f"\n   음성 설명: {st.session_state['voice_text']}"
    
    sbar = f"""[S] 환자 응급 이송 중. 증상: {symptom}. 위치: ({user_lat:.5f}, {user_lon:.5f}) / {st.session_state.get('addr') or ''}.{voice_detail}
[B] 기저/과거력: (현장 정보 입력 필요). 약물/알레르기: 미상.
[A] 실시간 설비/병상 기반 적합도 산정 → 후보 3곳 선정.
[R] 1순위: {top3.iloc[0]['dutyName']} (거리 {top3.iloc[0]['distance_km']:.2f}km, 점수 {top3.iloc[0]['score']}), 
    2순위: {top3.iloc[1]['dutyName'] if len(top3)>1 else '-'}, 
    3순위: {top3.iloc[2]['dutyName'] if len(top3)>2 else '-'}. 병원 대시보드/원무과에 사전통보 요망.
"""
    st.text_area("복사하여 의료진/원무과 공유", value=sbar, height=160)

# 자동 갱신 동작
if st.session_state["lat"] is not None and st.session_state["lon"] is not None:
    if auto_refresh:
        run_query_and_render()
    else:
        # 수동 실행 버튼 제공
        if st.button("🔄 수동으로 추천 갱신", type="primary", use_container_width=True):
            run_query_and_render()
else:
    # 위치가 없을 때 안내
    st.markdown("---")
    st.warning("### ⚠️ 위치 정보가 필요합니다")
    st.info("""
    **아래 단계를 따라 위치를 설정하세요:**
    
    1️⃣ 상단의 **"📍 내 위치 재설정"** 버튼을 클릭하세요
    
    2️⃣ 자동으로 위치를 가져올 수 없는 경우, 수동으로 입력할 수 있습니다
    
    3️⃣ 위치 설정 후 자동으로 근처 병원 정보가 표시됩니다
    """)
    
    # 간편 수동 입력
    st.markdown("### 📍 또는 여기서 바로 입력하세요")
    col_quick1, col_quick2, col_quick3 = st.columns([2, 2, 1])
    with col_quick1:
        quick_lat = st.number_input("위도 (예: 35.6)", value=35.6, format="%.6f", key="quick_lat")
    with col_quick2:
        quick_lon = st.number_input("경도 (예: 126.5)", value=126.5, format="%.6f", key="quick_lon")
    with col_quick3:
        st.write("")
        st.write("")
        if st.button("✅ 적용", type="primary", use_container_width=True):
            st.session_state["lat"] = quick_lat
            st.session_state["lon"] = quick_lon
            st.session_state["loc_source"] = "Manual"
            
            if KAKAO_KEY:
                addr = kakao_coord2address(quick_lon, quick_lat, KAKAO_KEY) or ""
                st.session_state["addr"] = addr
                reg = kakao_coord2region(quick_lon, quick_lat, KAKAO_KEY)
                if reg:
                    st.session_state["sido"], st.session_state["sigungu"], _ = reg
            
            st.success("✅ 위치 설정 완료!")
            st.rerun()

# 하단 주의 문구
st.info("※ 데모용 계산(적합도/룰)은 참고용이며, 실제 의료진 판단과 병원 응답이 최우선입니다. 데이터 누락/지연 가능.")
