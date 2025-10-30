
# er_triage_streamlit_app.py
# -*- coding: utf-8 -*-
import os, math
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse

import requests, pandas as pd, streamlit as st, pydeck as pdk
from geopy.distance import geodesic

# API 키는 Streamlit Secrets에서 로드 (로컬에서는 환경변수 사용)
DATA_GO_KR_KEY = st.secrets.get("DATA_GO_KR_SERVICE_KEY", os.getenv("DATA_GO_KR_SERVICE_KEY", ""))
KAKAO_KEY = st.secrets.get("KAKAO_REST_API_KEY", os.getenv("KAKAO_REST_API_KEY", ""))

ER_BED_URL = "https://apis.data.go.kr/B552657/ErmctInfoInqireService/getEmrrmRltmUsefulSckbdInfoInqire"
EGET_BASE_URL = "https://apis.data.go.kr/B552657/ErmctInfoInqireService/getEgytBassInfoInqire"
KAKAO_COORD2REGION_URL = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
KAKAO_COORD2ADDR_URL   = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
KAKAO_ADDRESS_URL      = "https://dapi.kakao.com/v2/local/search/address.json"

def _http_get(url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> requests.Response:
    timeout = (5, 15)
    params = dict(params) if params else {}
    if url.startswith("http://apis.data.go.kr/"):
        url = url.replace("http://", "https://", 1)
    netloc = urlparse(url).netloc
    if "apis.data.go.kr" in netloc:
        svc_key = params.pop("serviceKey", DATA_GO_KR_KEY)
        if svc_key:
            if "%" in svc_key:
                join = "&" if ("?" in url) else "?"
                url = f"{url}{join}serviceKey={svc_key}"
            else:
                params["serviceKey"] = svc_key
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp

def kakao_coord2region(lon: float, lat: float, kakao_key: str) -> Optional[Tuple[str, str, str]]:
    if not kakao_key: return None
    headers = {"Authorization": f"KakaoAK {kakao_key}"}
    params = {"x": lon, "y": lat}
    try:
        r = _http_get(KAKAO_COORD2REGION_URL, params=params, headers=headers)
        data = r.json(); docs = data.get("documents", [])
        target = next((d for d in docs if d.get("region_type")=="B"), docs[0] if docs else None)
        if not target: return None
        return target.get("region_1depth_name"), target.get("region_2depth_name"), target.get("code")
    except Exception:
        return None

def kakao_coord2address(lon: float, lat: float, kakao_key: str) -> Optional[str]:
    if not kakao_key: return None
    headers = {"Authorization": f"KakaoAK {kakao_key}"}
    params = {"x": lon, "y": lat}
    try:
        r = _http_get(KAKAO_COORD2ADDR_URL, params=params, headers=headers)
        data = r.json(); docs = data.get("documents", [])
        if not docs: return None
        d0 = docs[0]
        if d0.get("road_address") and d0["road_address"].get("address_name"):
            return d0["road_address"]["address_name"]
        if d0.get("address") and d0["address"].get("address_name"):
            return d0["address"]["address_name"]
        return None
    except Exception:
        return None

def kakao_address2coord(address: str, kakao_key: str) -> Optional[Tuple[float, float]]:
    if not kakao_key: return None
    headers = {"Authorization": f"KakaoAK {kakao_key}"}
    params = {"query": address}
    try:
        r = _http_get(KAKAO_ADDRESS_URL, params=params, headers=headers)
        data = r.json(); docs = data.get("documents", [])
        if not docs: return None
        first = docs[0]; lon = float(first["x"]); lat = float(first["y"])
        return (lat, lon)
    except Exception:
        return None

def fetch_baseinfo_by_hpid(hpid: str, service_key: str) -> Optional[Dict[str, Any]]:
    from xml.etree import ElementTree as ET
    try:
        r = _http_get(EGET_BASE_URL, {"HPID": hpid, "pageNo": 1, "numOfRows": 1, "serviceKey": service_key})
        root = ET.fromstring(r.content); it = root.find(".//item")
        if it is None: return None
        def g(tag):
            el = it.find(tag); return el.text.strip() if el is not None and el.text is not None else None
        return {
            "hpid": g("hpid"), "dutyName": g("dutyName") or g("dutyname"), "dutyAddr": g("dutyAddr"),
            "dutytel3": g("dutyTel3"),
            "wgs84Lat": float(g("wgs84Lat")) if g("wgs84Lat") else None,
            "wgs84Lon": float(g("wgs84Lon")) if g("wgs84Lon") else None,
        }
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_emergency_hospitals_in_region(sido: str, sigungu: str, service_key: str, max_items: int = 200) -> pd.DataFrame:
    from xml.etree import ElementTree as ET
    r = _http_get(ER_BED_URL, {"STAGE1": sido, "STAGE2": sigungu, "pageNo": 1, "numOfRows": 500, "serviceKey": service_key})
    root = ET.fromstring(r.content); hpids = []
    for it in root.findall(".//item"):
        el = it.find("hpid"); 
        if el is not None and el.text: hpids.append(el.text.strip())
    hpids = list(dict.fromkeys(hpids))[:max_items]
    rows = []
    for h in hpids:
        info = fetch_baseinfo_by_hpid(h, service_key)
        if info and info.get("wgs84Lat") and info.get("wgs84Lon"): rows.append(info)
    df = pd.DataFrame(rows)
    if not df.empty: df = df.dropna(subset=["wgs84Lat","wgs84Lon"])
    return df

@st.cache_data(ttl=30)
def fetch_er_beds(sido: str, sigungu: str, service_key: str, rows: int = 500) -> pd.DataFrame:
    from xml.etree import ElementTree as ET
    params = {"STAGE1": sido, "STAGE2": sigungu, "pageNo": 1, "numOfRows": rows, "serviceKey": service_key}
    resp = _http_get(ER_BED_URL, params=params); root = ET.fromstring(resp.content); items = root.findall(".//item")
    rows_list = []
    for it in items:
        def g(tag):
            el = it.find(tag); return el.text.strip() if el is not None and el.text is not None else None
        rows_list.append({
            "hpid": g("hpid"), "dutyName": g("dutyname"), "hvidate": g("hvidate"),
            "hvec": g("hvec"), "hvoc": g("hvoc"), "hvicc": g("hvicc"), "hvgc": g("hvgc"),
            "hvcc": g("hvcc"), "hvncc": g("hvncc"), "hvccc": g("hvccc"),
            "hvctayn": g("hvctayn"), "hvmriayn": g("hvmriayn"), "hvangioayn": g("hvangioayn"), "hvventiayn": g("hvventiayn"),
            "hv1": g("hv1"), "hv2": g("hv2"), "hv3": g("hv3"), "hv4": g("hv4"),
            "hv5": g("hv5"), "hv6": g("hv6"), "hv7": g("hv7"), "hv8": g("hv8"),
            "hv9": g("hv9"), "hv10": g("hv10"), "hv11": g("hv11"), "hv12": g("hv12"),
            "dutytel3": g("dutytel3") or g("hv1"), "hvdnm": g("hvdnm"),
        })
    df = pd.DataFrame(rows_list)
    if not df.empty: df = df.dropna(subset=["hpid"])
    return df

def add_distance_km(df: pd.DataFrame, user_lat: float, user_lon: float) -> pd.DataFrame:
    if df.empty:
        df["distance_km"] = []; return df
    distances = []
    for _, row in df.iterrows():
        try:
            lat, lon = float(row["wgs84Lat"]), float(row["wgs84Lon"])
            d = geodesic((user_lat, user_lon), (lat, lon)).km
        except Exception:
            d = math.nan
        distances.append(d)
    df = df.copy(); df["distance_km"] = distances
    return df.sort_values("distance_km")

def _safe_int(x):
    try: return int(str(x).strip()) if str(x).strip() not in ("", "None", "nan") else 0
    except: return 0

SYMPTOM_RULES = {
    "뇌졸중 의심(FAST+)": {"bool_any":[("hvctayn","Y")], "min_ge1":[("hvicc",1)], "nice_to_have":[("hv5",1),("hv6",1)]},
    "심근경색 의심(STEMI)": {"bool_any":[("hvangioayn","Y")], "min_ge1":[("hvoc",1),("hvicc",1)], "nice_to_have":[]},
    "다발성 외상/중증 외상": {"bool_any":[("hvventiayn","Y")], "min_ge1":[("hvoc",1),("hvicc",1)], "nice_to_have":[("hv9",1)]},
    "소아 중증(호흡곤란/경련 등)": {"bool_any":[("hv10","Y"),("hv11","Y")], "min_ge1":[("hvncc",1)], "nice_to_have":[]},
    "정형외과 중증(대형골절/절단)": {"bool_any":[], "min_ge1":[("hvoc",1),("hv3",1),("hv4",1)], "nice_to_have":[]},
    "신경외과 응급(의식저하/외상성출혈)": {"bool_any":[("hvctayn","Y")], "min_ge1":[("hv6",1),("hvicc",1)], "nice_to_have":[]},
}

def meets_requirements(row: pd.Series, rule: Dict[str, Any]) -> bool:
    for key, want in rule.get("bool_any", []):
        if str(row.get(key, "")).strip().upper() != want: return False
    for key, thr in rule.get("min_ge1", []):
        if _safe_int(row.get(key)) < thr: return False
    return True

def guess_region_from_address(addr: Optional[str]) -> Optional[Tuple[str, str]]:
    if not addr: return None
    parts = str(addr).strip().split()
    if len(parts)>=2: return parts[0], parts[1]
    return None

st.set_page_config(page_title="증상맞춤 응급 병상 Top3", page_icon="🚑", layout="wide")
st.title("🚑 증상 맞춤: 내 위치 기준 응급 병상 Top 3")
st.caption("• 데이터: 국립중앙의료원 응급의료 Open API / 카카오 로컬 • 데모 목적 — 실제 운용 전 기관 협의 및 데이터 검증 필요")

with st.expander("🔑 API 키 설정", expanded=not (DATA_GO_KR_KEY and KAKAO_KEY)):
    col_a, col_b = st.columns(2)
    with col_a:
        DATA_GO_KR_KEY = st.text_input("DATA_GO_KR_SERVICE_KEY (일반키 추천)", value=DATA_GO_KR_KEY, type="password")
    with col_b:
        KAKAO_KEY = st.text_input("KAKAO_REST_API_KEY", value=KAKAO_KEY, type="password")

st.divider()

for k, v in [("auto_lat", None), ("auto_lon", None), ("auto_addr", "")]:
    if k not in st.session_state: st.session_state[k] = v

st.subheader("📍 내 위치")
colA, colB = st.columns([3,2], vertical_alignment="bottom")
with colA:
    st.text_input("내 위치(주소)", key="auto_addr", placeholder="예: 광주광역시 동구 문화전당로 38")
with colB:
    if st.button("📍 내 위치 재설정(GPS)", use_container_width=True):
        lat = lon = None
        try:
            from streamlit_js_eval import get_geolocation
            loc = get_geolocation()
            if loc and isinstance(loc, dict):
                coords = loc.get("coords") or loc
                lat = coords.get("latitude"); lon = coords.get("longitude")
        except Exception: pass
        if lat is None or lon is None:
            try:
                from streamlit_geolocation import geolocation
                g = geolocation("📍 위치 권한을 허용해 주세요")
                if g and g.get("latitude") and g.get("longitude"):
                    lat = g["latitude"]; lon = g["longitude"]
            except Exception: pass
        if lat is not None and lon is not None:
            st.session_state["auto_lat"] = float(lat); st.session_state["auto_lon"] = float(lon)
            addr = kakao_coord2address(float(lon), float(lat), KAKAO_KEY) or ""
            st.session_state["auto_addr"] = addr
            st.success(f"현재 위치 설정 완료: lat={lat:.6f}, lon={lon:.6f}")
        else:
            st.error("브라우저 위치를 가져올 수 없습니다. HTTPS(또는 localhost)에서 위치 권한을 허용해 주세요.")

lat_show = st.session_state.get("auto_lat"); lon_show = st.session_state.get("auto_lon")
st.caption(f"현재 좌표: {lat_show if lat_show is not None else '—'}, {lon_show if lon_show is not None else '—'}")

recalc = st.button("주소로 좌표 재계산")
if recalc and st.session_state.get("auto_addr") and KAKAO_KEY:
    coord = kakao_address2coord(st.session_state["auto_addr"], KAKAO_KEY)
    if coord:
        st.session_state["auto_lat"], st.session_state["auto_lon"] = coord[0], coord[1]
        st.success(f"좌표 변환 성공: lat={coord[0]:.6f}, lon={coord[1]:.6f}")
    else:
        st.error("주소 → 좌표 변환 실패. 주소를 다시 확인하세요.")

user_lat = st.session_state.get("auto_lat"); user_lon = st.session_state.get("auto_lon")
if user_lat is None or user_lon is None:
    st.info("먼저 ‘📍 내 위치 재설정(GPS)’ 버튼을 누르거나, 주소를 입력하고 ‘주소로 좌표 재계산’을 눌러주세요."); st.stop()

region = kakao_coord2region(user_lon, user_lat, KAKAO_KEY) if KAKAO_KEY else None
guessed = None
if region:
    sido, sigungu, code = region
    st.write(f"행정구역: **{sido} {sigungu}** (code: {code})")
else:
    guessed = guess_region_from_address(st.session_state.get("auto_addr"))
    if guessed:
        sido, sigungu = guessed
        st.info(f"카카오 역지오코딩 실패 → 주소 문자열로 추정 사용: **{sido} {sigungu}**")
    else:
        st.warning("카카오 역지오코딩 실패 — 시/도, 시/군/구를 직접 입력하세요.")
        colr1, colr2 = st.columns(2)
        with colr1:  sido = st.text_input("시/도 (예: 광주광역시)", value="광주광역시")
        with colr2:  sigungu = st.text_input("시/군/구 (예: 동구)", value="동구")

with st.expander("🔍 진단(카카오 키·위치 확인)", expanded=False):
    st.write("KAKAO 키 설정 여부:", bool(KAKAO_KEY))
    st.write("현재 좌표:", user_lat, user_lon)
    try:
        st.write("coord2address:", kakao_coord2address(user_lon, user_lat, KAKAO_KEY))
    except Exception as e:
        st.error(f"coord2address 에러: {e}")
    try:
        st.write("coord2region:", kakao_coord2region(user_lon, user_lat, KAKAO_KEY))
    except Exception as e:
        st.error(f"coord2region 에러: {e}")

st.subheader("🩺 환자 증상 선택")
symptom = st.selectbox("지금 환자에게 가장 가까운 카테고리를 고르세요", list(SYMPTOM_RULES.keys()), index=0)

# 선택한 증상에 필요한 병상/장비 표시
rule = SYMPTOM_RULES.get(symptom, {})

# 필수 장비/시설 매핑
facility_names = {
    "hvctayn": "CT",
    "hvmriayn": "MRI",
    "hvangioayn": "조영촬영기",
    "hvventiayn": "인공호흡기",
    "hv10": "VENTI(소아)",
    "hv11": "인큐베이터",
}

bed_names = {
    "hvec": "응급실",
    "hvoc": "수술실",
    "hvicc": "일반중환자실",
    "hvncc": "신생중환자",
    "hvcc": "신경중환자",
    "hvccc": "흉부중환자",
    "hvgc": "입원실",
    "hv2": "내과중환자실",
    "hv3": "외과중환자실",
    "hv4": "외과입원실(정형외과)",
    "hv5": "신경과입원실",
    "hv6": "신경외과중환자실",
    "hv7": "약물중환자",
    "hv8": "화상중환자",
    "hv9": "외상중환자",
}

st.markdown(f"""
<div style="
    padding: 1rem; 
    border-left: 4px solid #0284c7; 
    background: #f0f9ff; 
    border-radius: 8px; 
    margin: 1rem 0;
">
    <p style="margin: 0.3rem 0; font-size: 1rem; color: #0c4a6e;"><b>📋 이 증상에 필요한 병원 시설:</b></p>
""", unsafe_allow_html=True)

# 필수 장비
required_facilities = [facility_names.get(k) for k, _ in rule.get("bool_any", []) if k in facility_names]
if required_facilities:
    st.markdown(f"<p style='margin: 0.3rem 0 0.3rem 1rem; font-size: 0.95rem;'>🔴 <b>필수 장비:</b> {', '.join(required_facilities)}</p>", unsafe_allow_html=True)

# 필수 병상
required_beds = [bed_names.get(k) for k, _ in rule.get("min_ge1", []) if k in bed_names]
if required_beds:
    st.markdown(f"<p style='margin: 0.3rem 0 0.3rem 1rem; font-size: 0.95rem;'>🔴 <b>필수 병상:</b> {', '.join(required_beds)}</p>", unsafe_allow_html=True)

# 권장 병상
nice_beds = [bed_names.get(k) for k, _ in rule.get("nice_to_have", []) if k in bed_names]
if nice_beds:
    st.markdown(f"<p style='margin: 0.3rem 0 0.3rem 1rem; font-size: 0.95rem;'>🟡 <b>권장 병상:</b> {', '.join(nice_beds)}</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

if st.button("지금 내 위치 기준 추천 3곳 보기", type="primary", use_container_width=True):
    if not DATA_GO_KR_KEY: st.error("DATA_GO_KR_SERVICE_KEY가 필요합니다."); st.stop()
    with st.spinner("병원 기본정보(좌표) 조회 중..."):
        hospitals = fetch_emergency_hospitals_in_region(sido, sigungu, DATA_GO_KR_KEY, max_items=200)
    if hospitals.empty: st.error("해당 행정구역의 응급 대상 병원을 찾지 못했습니다."); st.stop()

    with st.spinner("실시간 응급 병상/장비 조회 중..."):
        beds = fetch_er_beds(sido, sigungu, DATA_GO_KR_KEY, rows=500)

    all_merged = pd.merge(hospitals, beds, on="hpid", how="left", suffixes=("", "_bed"))
    if all_merged.empty: st.error("병원 기본정보와 병상 정보가 조인되지 않았습니다."); st.stop()

    rule = SYMPTOM_RULES.get(symptom, {})
    needed_cols = set([k for k,_ in rule.get("bool_any", [])] + [k for k,_ in rule.get("min_ge1", [])] +
                      ["dutytel3","hvdnm","dutyName","dutyAddr","hvec","hvoc","hvgc",
                       "hv1","hv2","hv3","hv4","hv5","hv6","hv7","hv8","hv9",
                       "hvicc","hvcc","hvncc","hvccc","hvidate"])
    for c in needed_cols:
        if c not in all_merged.columns: all_merged[c] = None

    eligible = all_merged[all_merged.apply(lambda r: meets_requirements(r, rule), axis=1)].copy()
    if not eligible.empty:
        eligible = add_distance_km(eligible, user_lat, user_lon)
        eligible["__fresh_m"] = eligible["hvidate"].map(lambda s: 0 if s else 9999)
        eligible = eligible.sort_values(by=["distance_km", "__fresh_m"], ascending=[True, True])
        top3 = eligible.head(3).copy()
    else:
        st.warning("증상 조건을 모두 만족하는 병원을 찾지 못했습니다. 가까운 순으로 대체 제안합니다.")
        relaxed = add_distance_km(all_merged, user_lat, user_lon)
        top3 = relaxed.sort_values("distance_km").head(3).copy()

    # 카카오 길찾기 API로 정확한 경로 및 소요 시간 계산
    def get_driving_info_kakao(origin_lat, origin_lon, dest_lat, dest_lon, kakao_key):
        """카카오 모빌리티 길찾기 API 호출 - 경로 좌표 포함"""
        if not kakao_key:
            return None, None, None
        
        url = "https://apis-navi.kakaomobility.com/v1/directions"
        headers = {"Authorization": f"KakaoAK {kakao_key}"}
        params = {
            "origin": f"{origin_lon},{origin_lat}",
            "destination": f"{dest_lon},{dest_lat}",
            "priority": "RECOMMEND",  # 추천 경로
        }
        
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                routes = data.get("routes", [])
                if routes:
                    route = routes[0]
                    summary = route.get("summary", {})
                    distance_m = summary.get("distance", 0)  # 미터
                    duration_sec = summary.get("duration", 0)  # 초
                    
                    distance_km = distance_m / 1000
                    duration_min = int(duration_sec / 60)
                    
                    # 실제 경로 좌표 추출
                    path_coords = []
                    sections = route.get("sections", [])
                    for section in sections:
                        roads = section.get("roads", [])
                        for road in roads:
                            for vertex in road.get("vertexes", []):
                                # vertexes는 [lon, lat, lon, lat, ...] 형식
                                pass
                            # 더 간단하게: section의 guides 사용
                        guides = section.get("guides", [])
                        for guide in guides:
                            x = guide.get("x")  # 경도
                            y = guide.get("y")  # 위도
                            if x and y:
                                path_coords.append([x, y])
                    
                    return distance_km, duration_min, path_coords
        except Exception as e:
            print(f"Kakao API error: {e}")
        
        return None, None, None
    
    # 각 병원에 대해 카카오 API로 경로 조회
    route_paths = {}  # 병원별 실제 경로 좌표 저장
    
    if KAKAO_KEY:
        with st.spinner("🚗 실제 경로 및 소요 시간 계산 중..."):
            for idx in top3.index:
                h_lat = top3.at[idx, "wgs84Lat"]
                h_lon = top3.at[idx, "wgs84Lon"]
                
                if h_lat and h_lon:
                    real_dist, real_eta, path_coords = get_driving_info_kakao(user_lat, user_lon, h_lat, h_lon, KAKAO_KEY)
                    
                    if real_dist and real_eta:
                        top3.at[idx, "distance_km"] = real_dist
                        top3.at[idx, "eta_minutes"] = real_eta
                        if path_coords:
                            route_paths[idx] = path_coords  # 경로 저장
                    else:
                        # API 실패 시 기존 추정값 사용
                        if isinstance(top3.at[idx, "distance_km"], (float, int)):
                            dist = top3.at[idx, "distance_km"]
                            top3.at[idx, "eta_minutes"] = int((dist * 1.3 / 40) * 60)
    
    if "distance_km" in top3.columns:
        top3["distance_km"] = top3["distance_km"].map(lambda x: f"{x:.2f} km" if isinstance(x,(float,int)) else x)
    
    # hvidate 포맷 변환: "20250130141500" → "2025-01-30 14:15"
    def format_hvidate(date_str):
        if not date_str or str(date_str).strip() in ("", "None", "nan"): return "없음"
        s = str(date_str).strip()
        if len(s) >= 12:
            return f"{s[0:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}"
        return s
    
    if "hvidate" in top3.columns:
        top3["hvidate"] = top3["hvidate"].map(format_hvidate)
    
    # 당직의명 + 전화번호 결합
    def format_doctor_info(row):
        doc_name = row.get("hvdnm")
        tel = row.get("dutytel3")
        if doc_name and str(doc_name).strip() not in ("", "None", "nan", "없음"):
            doc_name = str(doc_name).strip()
            if tel and str(tel).strip() not in ("", "None", "nan", "없음"):
                tel = str(tel).strip()
                return f"{doc_name} (☎️ {tel})"
            return doc_name
        return "없음"
    
    if "hvdnm" in top3.columns:
        top3["당직의정보"] = top3.apply(format_doctor_info, axis=1)
    
    # None 값을 "없음"으로 일괄 변환
    def replace_none(val):
        if val is None or str(val).strip() in ("None", "nan", ""): return "없음"
        return val
    
    for col in top3.columns:
        if col not in ["distance_km", "hvidate", "당직의정보"]:  # 이미 처리된 컬럼 제외
            top3[col] = top3[col].map(replace_none)

    st.subheader("🏆 증상 조건을 만족하는 가까운 병원 Top 3")
    
    for idx, (_, row) in enumerate(top3.iterrows(), 1):
        with st.container():
            # 헤더 박스
            eta = row.get('eta_minutes', 0)
            eta_text = f"약 {eta}분" if eta else "계산 중"
            st.markdown(f"""
            <div style="padding: 1.5rem; border: 3px solid #0284c7; border-radius: 12px; margin-bottom: 0.5rem; background: linear-gradient(to right, #f0f9ff, #e0f2fe); box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h2 style="color: #0c4a6e; margin: 0 0 1rem 0; font-size: 1.8rem;">🏥 {idx}위: {row['dutyName']}</h2>
                <p style="margin: 0.5rem 0; font-size: 1.3rem;"><b>📍 거리:</b> <span style="color: #dc2626; font-weight: bold;">{row['distance_km']}</span></p>
                <p style="margin: 0.3rem 0 0.5rem 2.5rem; font-size: 1.1rem; color: #0c4a6e;"><b>🚗 예상 소요시간:</b> <span style="color: #ea580c; font-weight: bold;">{eta_text}</span> <span style="font-size: 0.9rem; color: #64748b;">(자가용 기준)</span></p>
                <p style="margin: 0.5rem 0; font-size: 1.15rem;"><b>🏠 주소:</b> {row['dutyAddr']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 병상 정보 - 선택한 증상에 필요한 것만 표시
            st.markdown("### ✅ 이용 가능한 병상:")
            
            # 전체 병상 정보 매핑
            all_bed_mapping = {
                "hvec": ("🚨 응급실", row.get("hvec", "없음")),
                "hvoc": ("🏥 수술실", row.get("hvoc", "없음")),
                "hvgc": ("🏨 입원실", row.get("hvgc", "없음")),
                "hv2": ("💊 내과중환자실", row.get("hv2", "없음")),
                "hv3": ("🔪 외과중환자실", row.get("hv3", "없음")),
                "hv4": ("🦴 외과입원실(정형외과)", row.get("hv4", "없음")),
                "hv5": ("🧠 신경과입원실", row.get("hv5", "없음")),
                "hv6": ("🧠 신경외과중환자실", row.get("hv6", "없음")),
                "hvicc": ("⚕️ 일반중환자실", row.get("hvicc", "없음")),
                "hvcc": ("🧠 신경중환자", row.get("hvcc", "없음")),
                "hvncc": ("👶 신생중환자", row.get("hvncc", "없음")),
                "hvccc": ("🫁 흉부중환자", row.get("hvccc", "없음")),
                "hv7": ("💉 약물중환자", row.get("hv7", "없음")),
                "hv8": ("🔥 화상중환자", row.get("hv8", "없음")),
                "hv9": ("🚑 외상중환자", row.get("hv9", "없음")),
            }
            
            # 선택한 증상에 필요한 병상만 필터링
            needed_beds = set()
            for key, _ in rule.get("min_ge1", []):
                if key in all_bed_mapping:
                    needed_beds.add(key)
            for key, _ in rule.get("nice_to_have", []):
                if key in all_bed_mapping:
                    needed_beds.add(key)
            
            # 필요한 병상만 표시
            bed_items = [all_bed_mapping[key] for key in needed_beds if key in all_bed_mapping]
            
            # 있는 병상만 표시
            available = [(name, val) for name, val in bed_items if str(val) != "없음" and val]
            unavailable = [name for name, val in bed_items if str(val) == "없음" or not val]
            
            if available:
                cols = st.columns(min(len(available), 4))
                for i, (name, value) in enumerate(available):
                    with cols[i % 4]:
                        # Y/N 값은 "있음"으로, 숫자는 "N개"로 표시
                        if str(value).strip().upper() in ("Y", "N"):
                            display_value = "있음" if str(value).strip().upper() == "Y" else "없음"
                        else:
                            try:
                                num = int(value)
                                display_value = f"{num}개"
                            except:
                                display_value = str(value)
                        st.metric(name, display_value, delta=None)
            else:
                st.warning("⚠️ 현재 가용 병상 정보 없음")
            
            if unavailable:
                st.caption(f"미보유: {', '.join(unavailable)}")
            
            # 전화번호와 당직의 정보
            col_phone, col_update = st.columns([3, 2])
            with col_phone:
                tel = row.get("dutytel3")
                if tel and str(tel).strip() not in ("없음", "None", "nan", ""):
                    tel_clean = str(tel).strip()
                    # 발표용: 실제 연결은 010-2994-5413으로
                    demo_phone = "010-2994-5413"
                    st.markdown(f"""
                    <a href="tel:{demo_phone}" style="text-decoration: none;">
                        <button style="
                            background: #dc2626; 
                            color: white; 
                            padding: 1.2rem 2rem; 
                            border: none; 
                            border-radius: 10px; 
                            font-size: 1.3rem; 
                            font-weight: bold; 
                            cursor: pointer;
                            width: 100%;
                            box-shadow: 0 4px 8px rgba(220,38,38,0.3);
                            transition: all 0.3s;
                        " onmouseover="this.style.background='#b91c1c'" onmouseout="this.style.background='#dc2626'">
                            📞 응급실 바로 전화: {tel_clean}
                        </button>
                    </a>
                    """, unsafe_allow_html=True)
                    
                    # ARS 직통 버튼 (당직의 직통 연락처가 있는 경우)
                    direct_tel = row.get("hv1")  # 응급실 당직의 직통연락처
                    if direct_tel and str(direct_tel).strip() not in ("없음", "None", "nan", ""):
                        direct_tel_clean = str(direct_tel).strip()
                        st.markdown(f"""
                        <a href="tel:{demo_phone}" style="text-decoration: none;">
                            <button style="
                                background: #ea580c; 
                                color: white; 
                                padding: 0.8rem 1.5rem; 
                                border: none; 
                                border-radius: 8px; 
                                font-size: 1rem; 
                                font-weight: bold; 
                                cursor: pointer;
                                width: 100%;
                                margin-top: 0.5rem;
                                box-shadow: 0 2px 4px rgba(234,88,12,0.3);
                            " onmouseover="this.style.background='#c2410c'" onmouseout="this.style.background='#ea580c'">
                                📱 당직의 직통: 있음
                            </button>
                        </a>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="
                            background: #94a3b8; 
                            color: white; 
                            padding: 0.8rem 1.5rem; 
                            border-radius: 8px; 
                            font-size: 1rem; 
                            font-weight: bold; 
                            text-align: center;
                            width: 100%;
                            margin-top: 0.5rem;
                            opacity: 0.7;
                        ">
                            📱 당직의 직통: 없음
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("💡 연결 후 ARS 안내에 따라 응급실로 연결하세요")
                else:
                    st.info("📞 응급실 전화번호 정보 없음")
                
                # 당직의 정보 - 이름과 전화번호 분리 표시
                doc_name = row.get("hvdnm")
                doc_tel = row.get("hv1")
                
                if doc_name and str(doc_name).strip() not in ("없음", "None", "nan", ""):
                    doc_name_clean = str(doc_name).strip()
                    if doc_tel and str(doc_tel).strip() not in ("없음", "None", "nan", ""):
                        doc_tel_clean = str(doc_tel).strip()
                        st.markdown(f"""
                        <p style='font-size: 1.1rem; margin-top: 0.8rem;'>
                            👨‍⚕️ <b>당직의:</b> {doc_name_clean}<br/>
                            <span style='margin-left: 2rem; color: #0284c7;'>☎️ {doc_tel_clean}</span>
                        </p>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='font-size: 1.1rem; margin-top: 0.8rem;'>👨‍⚕️ <b>당직의:</b> {doc_name_clean}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='font-size: 1.1rem; margin-top: 0.8rem;'>👨‍⚕️ <b>당직의:</b> 없음</p>", unsafe_allow_html=True)
            
            with col_update:
                update_time = row.get("hvidate", "없음")
                st.metric("🕐 병상정보 업데이트", update_time)
            
            st.markdown("---")

    st.subheader("🗺️ 지도")
    
    # 순위별 색상
    rank_colors = [
        [220, 38, 38],    # 1위: 빨강
        [234, 88, 12],    # 2위: 주황
        [250, 204, 21],   # 3위: 노랑
    ]
    
    # 사용자 위치 (파란색)
    user_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": user_lat, "lon": user_lon, "color": [37, 99, 235]}],
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=100,
        pickable=False
    )
    
    # 병원 마커 & 경로선 데이터
    marker_data = []
    path_data = []  # 실제 도로 경로용
    
    for idx, (row_idx, r) in enumerate(top3.iterrows()):
        try:
            h_lat = float(r["wgs84Lat"])
            h_lon = float(r["wgs84Lon"])
            color = rank_colors[idx] if idx < 3 else [100, 100, 100]
            
            # 병원 마커
            eta = r.get("eta_minutes", 0)
            eta_text = f"약 {eta}분" if eta else "계산 중"
            marker_data.append({
                "lat": h_lat,
                "lon": h_lon,
                "name": r.get("dutyName"),
                "addr": r.get("dutyAddr"),
                "dist": r.get("distance_km"),
                "eta": eta_text,
                "color": color,
                "rank": idx + 1
            })
            
            # 실제 경로가 있으면 사용, 없으면 직선
            if row_idx in route_paths and route_paths[row_idx]:
                # 실제 도로 경로 좌표를 PathLayer용으로 변환
                path_coords = [[coord[0], coord[1]] for coord in route_paths[row_idx]]
                # 시작점 추가
                full_path = [[user_lon, user_lat]] + path_coords + [[h_lon, h_lat]]
                
                path_data.append({
                    "path": full_path,
                    "color": color + [200],  # 투명도
                    "width": 5
                })
            else:
                # API 실패 시 직선 경로 (대체)
                path_data.append({
                    "path": [[user_lon, user_lat], [h_lon, h_lat]],
                    "color": color + [150],
                    "width": 3
                })
        except Exception as e:
            print(f"Map data error: {e}")
            continue
    
    # 실제 경로 레이어 (PathLayer 사용)
    path_layer = pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        get_color="color",
        get_width="width",
        width_min_pixels=2,
        pickable=False
    )
    
    # 병원 마커 레이어
    hospital_layer = pdk.Layer(
        "ScatterplotLayer",
        data=marker_data,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=120,
        pickable=True
    )
    
    tooltip = {
        "html": "<b>🏥 {rank}위: {name}</b><br/>📍 {dist}<br/>🚗 {eta} (자가용)<br/>🏠 {addr}",
        "style": {"backgroundColor": "rgba(0,0,0,0.8)", "color": "white", "fontSize": "14px", "padding": "10px"}
    }
    
    # 지도 중심 계산
    mid_lat, mid_lon = (user_lat, user_lon)
    if marker_data:
        avg_lat = sum(m["lat"] for m in marker_data) / len(marker_data)
        avg_lon = sum(m["lon"] for m in marker_data) / len(marker_data)
        mid_lat = (user_lat + avg_lat) / 2
        mid_lon = (user_lon + avg_lon) / 2
    
    st.pydeck_chart(pdk.Deck(
        map_style=None,  # 기본 OpenStreetMap 스타일 (API 키 불필요)
        initial_view_state=pdk.ViewState(
            latitude=mid_lat,
            longitude=mid_lon,
            zoom=11.5,
            pitch=0
        ),
        layers=[path_layer, user_layer, hospital_layer],  # 실제 도로 경로 표시
        tooltip=tooltip
    ))
    
    # 범례
    st.markdown("""
    <div style="display: flex; gap: 1rem; margin-top: 0.5rem; font-size: 0.9rem;">
        <span>🔵 내 위치</span>
        <span style="color: #dc2626;">🔴 1위</span>
        <span style="color: #ea580c;">🟠 2위</span>
        <span style="color: #facc15;">🟡 3위</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("조회 완료!")
