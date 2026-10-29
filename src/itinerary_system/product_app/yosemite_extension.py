"""Three distinct Yosemite day routes with fixed, attributed visitor information."""

from __future__ import annotations

from copy import deepcopy
from itertools import permutations

from .workspace import WorkspaceError

YOSEMITE_POIS = {
    "yosemite_lower_fall": {
        "poi_id": "yosemite_lower_fall",
        "attraction_name": "Lower Yosemite Fall",
        "city": "Yosemite National Park",
        "latitude": 37.740588,
        "longitude": -119.601859,
        "visit_duration_minutes": 120,
        "official_url": "https://www.nps.gov/yose/planyourvisit/lowerfalltrail.htm",
        "source_list": "NPS YOSE access feature 5507",
        "guide": {
            "schema_version": "yosemite-stop-guide-v1",
            "theme": "山谷瀑布與草甸",
            "intro": "從公共停車區步行到下優勝美地瀑布步道，欣賞岩壁與瀑布。夏末或秋季可能水量很少；預留時間包含停車區往返。",
            "access_name": "Yosemite Falls Parking",
            "photo": {
                "url": "https://www.nps.gov/yose/planyourvisit/images/IMG_1192edit.jpg?maxwidth=650&autorotate=false",
                "alt": "Upper and Lower Yosemite Falls from the Lower Yosemite Fall trail",
                "caption_and_credit": "",
                "credit": "圖片來源：NPS 官方頁面；該圖未單獨列出攝影者",
                "source_url": "https://www.nps.gov/yose/planyourvisit/lowerfalltrail.htm",
            },
            "visitor_reviews": [
                {
                    "summary": "作者在行程中安排下瀑布短步行，之後步行前往 Cook’s Meadow。",
                    "author": "u/BBDBVAPA",
                    "source_url": "https://www.reddit.com/r/Yosemite/comments/1w3qqo3/6_days5_nights_at_yosemite_valley_lodge_follow_up/",
                    "source_title": "6 days/5 nights at Yosemite Valley Lodge — Trip Report",
                    "date_label": "讀取時顯示約一個月前；精確日期未核實",
                    "kind": "individual_visit_report",
                    "retrieved_on": "2026-10-20",
                }
            ],
            "review_status": "available_individual_report",
            "review_disclosure": "個人遊記摘述，並非整體評分；歷史經驗不代表目前狀況。",
        },
    },
    "yosemite_cooks_meadow": {
        "poi_id": "yosemite_cooks_meadow",
        "attraction_name": "Cook’s Meadow Loop",
        "city": "Yosemite National Park",
        "latitude": 37.744104,
        "longitude": -119.589824,
        "visit_duration_minutes": 75,
        "official_url": "https://www.nps.gov/yose/planyourvisit/cooksmeadowtrail.htm",
        "source_list": "NPS YOSE access feature 4190",
        "guide": {
            "schema_version": "yosemite-stop-guide-v1",
            "theme": "山谷瀑布與草甸",
            "intro": "從 Sentinel Bridge 停車區進入草甸與木棧道，從谷底欣賞 Half Dome 和 Yosemite Falls。沿正式步道行走，不進入草甸。",
            "access_name": "Sentinel Bridge Parking",
            "photo": {
                "url": "https://www.nps.gov/yose/planyourvisit/images/IMG_6693edit.jpg?maxwidth=650&autorotate=false",
                "alt": "Morning view of Yosemite Falls from Cook's Meadow",
                "caption_and_credit": "Morning view of Yosemite Falls from Cook's Meadow",
                "credit": "圖片來源：NPS 官方頁面；該圖未單獨列出攝影者",
                "source_url": "https://www.nps.gov/yose/planyourvisit/cooksmeadowtrail.htm",
            },
            "visitor_reviews": [
                {
                    "summary": "作者特別喜歡 Cook’s Meadow 的日落景色。",
                    "author": "u/BBDBVAPA",
                    "source_url": "https://www.reddit.com/r/Yosemite/comments/1w3qqo3/6_days5_nights_at_yosemite_valley_lodge_follow_up/",
                    "source_title": "6 days/5 nights at Yosemite Valley Lodge — Trip Report",
                    "date_label": "讀取時顯示約一個月前；精確日期未核實",
                    "kind": "individual_visit_report",
                    "retrieved_on": "2026-10-20",
                }
            ],
            "review_status": "available_individual_report",
            "review_disclosure": "個人遊記摘述，並非整體評分；歷史經驗不代表目前狀況。",
        },
    },
    "yosemite_glacier_point": {
        "poi_id": "yosemite_glacier_point",
        "attraction_name": "Glacier Point",
        "city": "Yosemite National Park",
        "latitude": 37.726956,
        "longitude": -119.575298,
        "visit_duration_minutes": 90,
        "official_url": "https://www.nps.gov/yose/planyourvisit/glacierpoint.htm",
        "source_list": "NPS YOSE access feature 22941",
        "guide": {
            "schema_version": "yosemite-stop-guide-v1",
            "theme": "高處觀景與花崗岩步道",
            "intro": "從 Glacier Point 停車區步行到觀景台，俯看優勝美地山谷與 Half Dome。Glacier Point Road 季節性開放，實際日期須重查。",
            "access_name": "Glacier Point Parking",
            "photo": {
                "url": "https://www.nps.gov/yose/planyourvisit/images/400_3_GlacierPoint_JTrust_gp-web.jpg?maxwidth=650&maxheight=1300&autorotate=false&format=webp",
                "alt": "People looking out from Glacier Point towards Half Dome.",
                "caption_and_credit": "People looking out from Glacier Point towards Half Dome.",
                "credit": "圖片來源：NPS 官方頁面；該圖未單獨列出攝影者",
                "source_url": "https://www.nps.gov/yose/planyourvisit/glacierpoint.htm",
            },
            "visitor_reviews": [
                {
                    "summary": "作者到訪 Glacier Point 的經典觀景點並拍照，表示現場景色仍讓他印象深刻。",
                    "author": "u/BBDBVAPA",
                    "source_url": "https://www.reddit.com/r/Yosemite/comments/1w3qqo3/6_days5_nights_at_yosemite_valley_lodge_follow_up/",
                    "source_title": "6 days/5 nights at Yosemite Valley Lodge — Trip Report",
                    "date_label": "讀取時顯示約一個月前；精確日期未核實",
                    "kind": "individual_visit_report",
                    "retrieved_on": "2026-10-20",
                }
            ],
            "review_status": "available_individual_report",
            "review_disclosure": "個人遊記摘述，並非整體評分；歷史經驗不代表目前狀況。",
        },
    },
    "yosemite_sentinel_dome": {
        "poi_id": "yosemite_sentinel_dome",
        "attraction_name": "Sentinel Dome",
        "city": "Yosemite National Park",
        "latitude": 37.712552,
        "longitude": -119.586349,
        "visit_duration_minutes": 180,
        "official_url": "https://www.nps.gov/places/000/sentinel-dome-and-taft-point-trailhead.htm",
        "source_list": "NPS YOSE access feature 17864",
        "guide": {
            "schema_version": "yosemite-stop-guide-v1",
            "theme": "高處觀景與花崗岩步道",
            "intro": "從官方步道口步行登上 Sentinel Dome。官方估計約 1–2 小時；此處另留休息與停車緩衝。雷雨時不要登上裸露岩頂。",
            "access_name": "Sentinel Dome Trailhead (Pohono Trail)",
            "photo": {
                "url": "https://www.nps.gov/common/uploads/cropped_image/primary/A938EC55-09C5-1BBB-46D918531492E553.jpg?width=650&quality=90&mode=crop",
                "alt": "Pink and blue skies over sweeping views of granite cliffs and mountain peaks",
                "caption_and_credit": "View from the top of Sentinel Dome at sunset",
                "credit": "圖片來源：NPS 官方頁面；該圖未單獨列出攝影者",
                "source_url": "https://www.nps.gov/places/000/sentinel-dome-and-taft-point-trailhead.htm",
            },
            "visitor_reviews": [
                {
                    "summary": "作者分享登上 Sentinel Dome 的體驗，之後走向 Taft Point。",
                    "author": "u/BBDBVAPA",
                    "source_url": "https://www.reddit.com/r/Yosemite/comments/1w3qqo3/6_days5_nights_at_yosemite_valley_lodge_follow_up/",
                    "source_title": "6 days/5 nights at Yosemite Valley Lodge — Trip Report",
                    "date_label": "讀取時顯示約一個月前；精確日期未核實",
                    "kind": "individual_visit_report",
                    "retrieved_on": "2026-10-20",
                }
            ],
            "review_status": "available_individual_report",
            "review_disclosure": "個人遊記摘述，並非整體評分；歷史經驗不代表目前狀況。",
        },
    },
    "yosemite_mariposa_grove": {
        "poi_id": "yosemite_mariposa_grove",
        "attraction_name": "Mariposa Grove",
        "city": "Yosemite National Park",
        "latitude": 37.506623,
        "longitude": -119.629246,
        "visit_duration_minutes": 360,
        "official_url": "https://www.nps.gov/yose/planyourvisit/mg.htm",
        "source_list": "NPS YOSE access feature 25316",
        "guide": {
            "schema_version": "yosemite-stop-guide-v1",
            "theme": "巨杉林與 Wawona 歷史",
            "intro": "開車停在 Welcome Plaza，使用當日開放的免費接駁或正式步道進入巨杉林。預留時間包含交通轉換、步行、休息；不把受限的 Grove Road 當作一般自駕道路。",
            "access_name": "Mariposa Grove Welcome Plaza and Parking",
            "photo": {
                "url": "https://www.nps.gov/yose/planyourvisit/images/mg_web.jpg?maxwidth=650&maxheight=1300&autorotate=false&format=webp",
                "alt": "Bachelor and Three Graces in the Mariposa Grove with sign",
                "caption_and_credit": "",
                "credit": "圖片來源：NPS 官方頁面；該圖未單獨列出攝影者",
                "source_url": "https://www.nps.gov/yose/planyourvisit/mg.htm",
            },
            "visitor_reviews": [
                {
                    "summary": "作者認為從南門進入時，順路走 Grizzly Giant 環線值得停留。",
                    "author": "u/BBDBVAPA",
                    "source_url": "https://www.reddit.com/r/Yosemite/comments/1w3qqo3/6_days5_nights_at_yosemite_valley_lodge_follow_up/",
                    "source_title": "6 days/5 nights at Yosemite Valley Lodge — Trip Report",
                    "date_label": "讀取時顯示約一個月前；精確日期未核實",
                    "kind": "individual_visit_report",
                    "retrieved_on": "2026-10-20",
                }
            ],
            "review_status": "available_individual_report",
            "review_disclosure": "個人遊記摘述，並非整體評分；歷史經驗不代表目前狀況。",
        },
    },
    "yosemite_wawona_history": {
        "poi_id": "yosemite_wawona_history",
        "attraction_name": "Yosemite History Center · Wawona",
        "city": "Yosemite National Park",
        "latitude": 37.537922,
        "longitude": -119.656765,
        "visit_duration_minutes": 60,
        "official_url": "https://www.nps.gov/yose/learn/historyculture/historycenter.htm",
        "source_list": "NPS YOSE access feature 23998",
        "guide": {
            "schema_version": "yosemite-stop-guide-v1",
            "theme": "巨杉林與 Wawona 歷史",
            "intro": "在 Wawona 的歷史建築與廊橋區步行，了解園區早期交通與居民故事。部分建築與示範活動只在季節或指定時間開放。",
            "access_name": "Yosemite History Center and Picnic Area Parking",
            "photo": {
                "url": "https://www.nps.gov/yose/learn/historyculture/images/history-center_1.jpg?maxwidth=650&maxheight=1300&autorotate=false&format=webp",
                "alt": "image of stage ride through history center and blacksmiths working in blacksmith shop",
                "caption_and_credit": "Left: Yosemite Transportation Office, present day (open in summer); Right: Blacksmith Shop, present day (open in summer)",
                "credit": "圖片來源：NPS 官方頁面；該圖未單獨列出攝影者",
                "source_url": "https://www.nps.gov/yose/learn/historyculture/historycenter.htm",
            },
            "visitor_reviews": [],
            "review_status": "No verifiable visitor review retained; not a zero rating",
            "review_disclosure": "個人遊記摘述，並非整體評分；歷史經驗不代表目前狀況。",
        },
    },
}
YOSEMITE_URLS = {
    "yosemite_lower_fall": "https://www.nps.gov/yose/planyourvisit/lowerfalltrail.htm",
    "yosemite_cooks_meadow": "https://www.nps.gov/yose/planyourvisit/cooksmeadowtrail.htm",
    "yosemite_glacier_point": "https://www.nps.gov/yose/planyourvisit/glacierpoint.htm",
    "yosemite_sentinel_dome": "https://www.nps.gov/places/000/sentinel-dome-and-taft-point-trailhead.htm",
    "yosemite_mariposa_grove": "https://www.nps.gov/yose/planyourvisit/mg.htm",
    "yosemite_wawona_history": "https://www.nps.gov/yose/learn/historyculture/historycenter.htm",
    "yosemite_conditions": "https://www.nps.gov/yose/planyourvisit/conditions.htm",
}

GROUPS = (
    ("yosemite_lower_fall", "yosemite_cooks_meadow"),
    ("yosemite_glacier_point", "yosemite_sentinel_dome"),
    ("yosemite_mariposa_grove", "yosemite_wawona_history"),
)
NOTICE = "Three added Yosemite days use distinct stops and themes. Access roads and hotel returns can overlap. Photos are from NPS; comments are attributed individual reports, not ratings. Recheck seasonal roads, shuttle hours and weather for the actual trip dates."


def extend_yosemite_solution(solution, matrix, max_drive):
    days = deepcopy(solution["days"])
    position = next((i for i, d in enumerate(days) if "yosemite_national_park" in d["stop_ids"]), None)
    if len(days) != 7 or position is None or position == 6:
        raise WorkspaceError("yosemite_extension_requires_seven_day_parent", 409)
    base = days[position]["end_anchor_id"]
    inserted = []
    for offset, group in enumerate(GROUPS, 1):
        candidates = []
        for order in permutations(group):
            ids = [base, *order, base]
            travel = sum(matrix.cell(u, v).require_duration_s() / 60 for u, v in zip(ids, ids[1:], strict=False))
            if travel <= max_drive and travel + sum(YOSEMITE_POIS[s]["visit_duration_minutes"] for s in order) <= 720:
                candidates.append((travel, order))
        if not candidates:
            raise WorkspaceError("yosemite_extension_day_infeasible", 409)
        _, order = min(candidates)
        inserted.append(
            {"day": position + 1 + offset, "start_anchor_id": base, "end_anchor_id": base, "stop_ids": list(order)}
        )
    for day in days[position + 1 :]:
        day["day"] += 3
    return {
        **solution,
        "days": [*days[: position + 1], *inserted, *days[position + 1 :]],
        "status": "FEASIBLE",
        "best_bound": None,
        "optimality_gap": None,
        "scope": "Seven-day finite-catalog MILP followed by exact ordering within three distinct Yosemite day groups; no joint ten-day global optimum claimed",
        "yosemite_added_days": 3,
        "daily_orders_checked": 6,
    }


def validate_yosemite_extension(plan, days, enabled):
    added = [s for s in plan.selected_stops if (s.get("poi_id") or s.get("stop_id")) in YOSEMITE_POIS]
    if not enabled:
        if added:
            raise WorkspaceError("yosemite_extension_request_mismatch", 409)
        return
    if len(days) != 10 or len(added) != 6 or {s["poi_id"] for s in added} != set(YOSEMITE_POIS):
        raise WorkspaceError("yosemite_extension_request_mismatch", 409)
    original = next(
        (s for s in plan.selected_stops if (s.get("poi_id") or s.get("stop_id")) == "yosemite_national_park"), None
    )
    if original is None:
        raise WorkspaceError("yosemite_extension_request_mismatch", 409)
    for offset, group in enumerate(GROUPS, 1):
        day_number = original["day"] + offset
        row = next((d for d in days if d["day"] == day_number), None)
        if row is None or set(row["stop_ids"]) != set(group) or row["start_anchor_id"] != row["end_anchor_id"]:
            raise WorkspaceError("yosemite_extension_day_mismatch", 409)
        for key in group:
            stop = next(s for s in added if s["poi_id"] == key)
            expected = YOSEMITE_POIS[key]
            if stop.get("day") != day_number or any(
                stop.get(k) != expected[k] for k in ("latitude", "longitude", "visit_duration_minutes", "guide")
            ):
                raise WorkspaceError("yosemite_extension_content_mismatch", 409)
