"""Typed requests for bounded California trip replanning."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import CopilotContextV2, CopilotIntentV1, CopilotIntentV2, CopilotInterpretationV1, CopilotInterpretationV2

TRIP_REPLAN = "replan_trip"
CITY_ANCHORS = {
    "San Francisco": "san_francisco_international_airport",
    "Los Angeles": "los_angeles_international_airport",
}
PLACE_ALIASES = {
    "redwood_national_park": (
        "Redwood National Park",
        "Red Wood National Park",
        "Redwoods National Park",
        "Redwood National and State Parks",
        "紅木國家公園",
        "红木国家公园",
    ),
    "channel_islands": ("Channel Islands", "海峽群島", "海峡群岛"),
    "hearst_castle": ("Hearst Castle", "赫斯特城堡", "赫氏古堡", "赫斯特古堡"),
    "kings_canyon_national_park": ("Kings Canyon", "國王峽谷", "国王峡谷"),
    "yosemite_national_park": ("Yosemite", "優勝美地", "优胜美地"),
    "sequoia_national_park": ("Sequoia", "紅杉國家公園", "红杉国家公园"),
    "bixby_creek_bridge_viewpoint": ("Bixby", "比克斯比大橋", "比克斯比大桥"),
    "point_lobos_state_natural_reserve": ("Point Lobos", "羅伯斯角", "罗伯斯角"),
    "pfeiffer_big_sur_state_park": ("Pfeiffer Big Sur",),
}
TRIP_PLACE_IDS = frozenset(PLACE_ALIASES) | {
    "golden_gate_bridge",
    "stanford_university_main_quad",
    "stearns_wharf",
    "griffith_observatory",
    "hollywood_walk_of_fame",
    "tcl_chinese_theatre",
    "santa_barbara_county_courthouse",
}


class TripPlanParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    start_city: Literal["San Francisco", "Los Angeles"] | None = None
    coastal: bool = False
    required_place_ids: list[str] = Field(default_factory=list, max_length=8)
    excluded_place_ids: list[str] = Field(default_factory=list, max_length=8)
    avoid_regions: list[Literal["big_sur"]] = Field(default_factory=list, max_length=1)
    verify_official: bool = False
    max_daily_drive_minutes: int = Field(default=720, ge=60, le=720)

    @model_validator(mode="after")
    def coherent(self) -> TripPlanParameters:
        for values in (self.required_place_ids, self.excluded_place_ids, self.avoid_regions):
            if len(values) != len(set(values)) or any(not re.fullmatch(r"[a-z][a-z0-9_]{0,95}", v) for v in values):
                raise ValueError("trip_planning_ids_invalid")
        if set(self.required_place_ids) & set(self.excluded_place_ids):
            raise ValueError("trip_planning_required_excluded_conflict")
        if not (set(self.required_place_ids) | set(self.excluded_place_ids)) <= TRIP_PLACE_IDS:
            raise ValueError("trip_planning_place_not_supported")
        if self.avoid_regions and not self.verify_official:
            raise ValueError("trip_planning_official_check_required")
        return self


class YosemiteExtensionParameters(TripPlanParameters):
    additional_yosemite_days: Literal[3]

    @model_validator(mode="after")
    def extension_coherent(self):
        if not self.verify_official or "yosemite_national_park" not in self.required_place_ids:
            raise ValueError("yosemite_extension_requires_official_and_original_visit")
        return self


class IndividualizedTripParameters(TripPlanParameters):
    total_days: int = Field(ge=1, le=21)
    pace: Literal["relaxed", "balanced", "active"] = "balanced"
    interests: list[Literal["nature", "culture", "coast"]] = Field(default_factory=list, max_length=3)
    profile_updates: list[Literal["pace", "interests", "max_daily_drive_minutes", "coastal"]] = Field(
        default_factory=list, max_length=4
    )

    @model_validator(mode="after")
    def unique_interests(self):
        if len(self.interests) != len(set(self.interests)) or len(self.profile_updates) != len(
            set(self.profile_updates)
        ):
            raise ValueError("trip_planning_interests_invalid")
        return self


def trip_day_count(parameters, parent):
    if isinstance(parameters, IndividualizedTripParameters):
        return parameters.total_days
    return max(s["day"] for s in parent["selected_stops"]) + getattr(parameters, "additional_yosemite_days", 0)


def resolve_trip_profile(requested, baseline, *, parent=None):
    previous = baseline.get("effective_parameters", baseline.get("parameters", {}))
    if not isinstance(requested, IndividualizedTripParameters):
        updates = (["coastal"] if requested.coastal else []) + (
            ["max_daily_drive_minutes"] if requested.max_daily_drive_minutes != 720 else []
        )
        inherited = {
            key: previous[key]
            for key in ("coastal", "max_daily_drive_minutes")
            if key in previous and key not in updates
        }
        inherited["excluded_place_ids"] = sorted(
            (set(previous.get("excluded_place_ids", [])) | set(requested.excluded_place_ids))
            - set(requested.required_place_ids)
        )
        if isinstance(requested, YosemiteExtensionParameters):
            if previous.get("pace", "balanced") != "balanced" or previous.get("interests"):
                from .workspace import WorkspaceError

                raise WorkspaceError("trip_planning_extension_profile_requires_clarification", 409)
            return requested.model_copy(update=inherited)
        if "total_days" not in previous:
            return requested.model_copy(update=inherited)
        return IndividualizedTripParameters.model_validate(
            {
                **requested.model_dump(),
                **inherited,
                "total_days": trip_day_count(requested, parent) if parent else previous["total_days"],
                "pace": previous.get("pace", "balanced"),
                "interests": previous.get("interests", []),
                "profile_updates": updates,
            }
        )
    inherited = {
        key: previous[key]
        for key in ("pace", "interests", "max_daily_drive_minutes", "coastal")
        if key in previous and key not in requested.profile_updates
    }
    inherited["excluded_place_ids"] = sorted(
        (set(previous.get("excluded_place_ids", [])) | set(requested.excluded_place_ids))
        - set(requested.required_place_ids)
    )
    return IndividualizedTripParameters.model_validate({**requested.model_dump(), **inherited})


def _number(text):
    if text.isdecimal():
        return int(text)
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "fourteen": 14,
        "twenty-one": 21,
    }
    if text.lower() in words:
        return words[text.lower()]
    digits = {
        "零": 0,
        "一": 1,
        "二": 2,
        "兩": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if "十" in text:
        left, right = text.split("十", 1)
        return digits.get(left, 1) * 10 + digits.get(right, 0)
    return digits.get(text)


def validate_trip_parameters(parameters: Any) -> TripPlanParameters:
    if type(parameters) is not dict:
        raise ValueError("trip_planning_parameters_invalid")
    model = (
        YosemiteExtensionParameters
        if "additional_yosemite_days" in parameters
        else IndividualizedTripParameters
        if "total_days" in parameters
        else TripPlanParameters
    )
    parsed = model.model_validate(parameters)
    if parsed.model_dump() != parameters:
        raise ValueError("trip_planning_parameters_incomplete")
    return parsed


def trip_request_interpretation(context: Any, message: str) -> CopilotInterpretationV1 | None:
    if not isinstance(context, CopilotContextV2) or not context.capabilities.get("operations", {}).get(
        TRIP_REPLAN, {}
    ).get("enabled"):
        return None
    from .copilot import request_edit_text

    message = request_edit_text(message)
    coastal = bool(
        re.search(
            r"(?:更多|集中|偏[向好]|改[成為为]|換成|换成|沿著|沿着|以).{0,12}海岸[線线]"
            r"|\bcoastal\s+(?:route|trip|itinerary|base)\b"
            r"|\b(?:focus|concentrate|prefer).{0,45}\bcoast(?:line|al)?\b"
            r"|\b(?:make|change|modify|replan|switch).{0,45}\bcoastal\b",
            message,
            re.I,
        )
    )
    official = bool(re.search(r"官[網网]|official|government website", message, re.I))
    big_sur = bool(re.search(r"big\s*sur|大蘇爾|大苏尔", message, re.I))
    start = re.search(
        r"(?:從|从)\s*(洛杉[磯矶]|舊金山|旧金山)\s*出[發发]|\bstart(?:ing)?(?:\s+(?:my|the|our)\s+trip)?\s+(?:from|in)\s+(los angeles|la|san francisco|sf)\b",
        message,
        re.I,
    )
    hearst = any(alias.casefold() in message.casefold() for alias in PLACE_ALIASES["hearst_castle"])
    islands = any(a.casefold() in message.casefold() for a in PLACE_ALIASES["channel_islands"])
    redwood = any(a.casefold() in message.casefold() for a in PLACE_ALIASES["redwood_national_park"])
    yosemite = any(a.casefold() in message.casefold() for a in PLACE_ALIASES["yosemite_national_park"])
    required_mentions = set()
    for clause in re.finditer(r"(?:\b(?:include|visit|add)\b|想去|必去|去)\s*([^.;!?。！？]*)", message, re.I):
        if re.search(r"不(?:是)?(?:很)?(?:想)?$", message[: clause.start()]):
            continue
        positive = re.split(
            r"\b(?:but|instead of|avoid|exclude|skip)\b|但|避開|避开|不要", clause[1], maxsplit=1, flags=re.I
        )[0]
        required_mentions.update(
            key
            for key, aliases in PLACE_ALIASES.items()
            if any(alias.casefold() in positive.casefold() for alias in aliases)
        )
    extension = yosemite and bool(
        re.search(r"(?:多|加|增|延[長长]|extra|more|extend|add).{0,35}(?:三|3|three)\s*(?:天|days?)", message, re.I)
    )
    number = (
        r"\d{1,3}|[一二兩两三四五六七八九十]{1,3}|twenty-one|fourteen|three|seven|eight|nine|four|five|six|one|two|ten"
    )
    duration_text = re.sub(rf"\b({number})\s+(?:more|extra|additional)\s+days?\b", r"\1 days", message, flags=re.I)
    day_matches = list(
        re.finditer(
            rf"(?<![第哪的每那這这\d一二兩两三四五六七八九十])({number})\s*(天|週|周|星期|weeks?|[- ]?days?)",
            duration_text,
            re.I,
        )
    )
    total_days = None
    if not extension and day_matches:
        values = {_number(m[1]) * (7 if re.fullmatch(r"週|周|星期|weeks?", m[2], re.I) else 1) for m in day_matches}
        transition = re.search(rf"(?:改成|改為|改为|變成|变成|\bto\s+)\s*({number})\s*(天|days?)", message, re.I)
        if transition:
            values = {_number(transition[1])}
            if (
                yosemite
                and values == {max(context.allowed_days) + 3}
                and re.search(r"新增的時間都留給|新增的时间都留给|added time.*Yosemite", message, re.I)
            ):
                return None
        if len(values) == 1:
            total_days = values.pop()
            if re.search(
                rf"(?:多|增加|新增|延長|延长)\s*({number})\s*天|(?:add|extend).{{0,20}}\b({number})\s+(?:(?:more|extra|additional)\s+)?days?|\b({number})\s+(?:more|extra|additional)\s+days?",
                message,
                re.I,
            ):
                total_days += max(context.allowed_days)
    drive_number = rf"\d{{1,3}}(?:\.\d{{1,2}})?|{number}"
    drive_match = re.search(
        rf"(?:每天|每日).{{0,5}}(?:開車|开车|駕駛|驾驶).{{0,12}}?(?<![\d.])({drive_number})(?![\d.])\s*(小時|小时|分鐘|分钟)|(?:drive|driving).{{0,25}}?(?<![\d.])({drive_number})(?![\d.])\s*(hours?|minutes?).{{0,15}}(?:per day|daily|a day)|\bdaily\s+driving\s+(?:under|limit\s+of)\s*(?<![\d.])({drive_number})(?![\d.])\s*(hours?|minutes?)\b",
        message,
        re.I,
    )
    comparative_driving = bool(
        re.search(
            r"\b(?:less|reduce|reduced|minimal|shorter)\s+(?:total\s+)?driving\b|\bdrive\s+less\b|"
            r"少(?:一點|一点)?(?:開車|开车|駕駛|驾驶)|減少.{0,6}(?:開車|开车|駕駛|驾驶)",
            message,
            re.I,
        )
    )
    pace = (
        "relaxed"
        if re.search(r"輕鬆|轻松|慢遊|慢游|relaxed|slow pace", message, re.I)
        else "active"
        if re.search(r"緊湊|紧凑|充實|充实|active pace|packed", message, re.I)
        else "balanced"
    )
    pace_specified = bool(
        re.search(
            r"輕鬆|轻松|慢遊|慢游|relaxed|slow pace|緊湊|紧凑|充實|充实|active pace|packed|平衡|balanced", message, re.I
        )
    )
    interests = [
        key
        for key, pattern in (
            ("nature", r"自然|森林|大自然|nature"),
            ("culture", r"文化|博物館|博物馆|歷史|历史|culture|museum"),
            ("coast", r"海岸|海邊|海边|coast|beach"),
        )
        if re.search(pattern, message, re.I)
    ]
    profile_request = bool(re.search(r"偏好|喜歡|喜欢|偏愛|偏爱|prefer|interest|I like", message, re.I))
    individualized = not extension and (
        bool(day_matches) or bool(drive_match) or pace_specified or bool(interests) and profile_request
    )
    if not (
        coastal
        or start
        or hearst
        or islands
        or redwood
        or extension
        or individualized
        or required_mentions
        or comparative_driving
        or (official and big_sur)
    ):
        return None
    chinese = bool(re.search(r"[\u3400-\u9fff]", message))

    def clarify(question: str | None = None) -> CopilotInterpretationV1:
        question = question or (
            "這條請求包含尚未能完整表示的條件。請分開說明整趟路線條件和局部編輯；目前沒有加入草稿。"
            if chinese
            else "I cannot represent every clause as one trip-planning request. Separate the trip requirements from local edits; nothing was staged."
        )
        return CopilotInterpretationV1(
            state="clarification_required",
            assistant_message=question,
            intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
        )

    comparative_required = bool(
        re.search(
            r"\b(?:must|guarantee|ensure|required)\b.{0,40}(?:less|reduc|shorter|driv)|"
            r"\b(?:less|shorter).{0,30}\bthan\b|一定|必須|必须|保證|保证|比.{0,16}(?:少|短)",
            message,
            re.I,
        )
    )
    if comparative_driving and (not drive_match or comparative_required):
        return clarify(
            ("已理解景點興趣為 " + ", ".join(interests) + "。" if interests else "")
            + "你希望少開車；請把完整需求連同每天最多開車幾小時或幾分鐘一起告訴我。我可以檢查每日上限，但不能保證總駕車時間比原行程更少。尚未建立部分草稿。"
            if chinese
            else ("I understood the interests: " + ", ".join(interests) + ". " if interests else "")
            + "You asked for less driving. Please restate the complete request with a maximum daily driving time in hours or minutes. "
            "Guaranteed reductions in total driving are not supported; I can check that daily cap. No partial draft was created."
        )
    if re.search(r"\b(?:replace|swap|substitute)\b|換成|换成|替換|替换", message, re.I):
        return clarify()
    if re.search(
        r"\b(?:tell me about|explain|where (?:do|can|is)|how (?:long|far|do|can)|what (?:is|are|can))\b|^\s*(?:is|are|does|do|which)\b|介紹一下|介绍一下|在哪裡|在哪里|多久|哪一天|哪天",
        message,
        re.I,
    ):
        if re.fullmatch(r"\s*What is Hearst Castle\s*[?!.]?\s*", message, re.I):
            from .official_travel import OFFICIAL_URLS

            return CopilotInterpretationV1(
                state="unsupported",
                assistant_message=(
                    "I do not have a registered reference description of Hearst Castle in this planner. "
                    f"For information about the site, see its [official FAQ]({OFFICIAL_URLS['hearst_castle']}). "
                    "Current opening hours, tours and tickets have not been checked. "
                    "Nothing was staged or changed."
                ),
                intents=(CopilotIntentV1(type="unsupported_request"),),
            )
        return clarify()
    if re.search(
        r"\b(?:do not|don't|don’t|not|never)\s+(?:like|prefer|want|interested)|\b(?:hate|dislike|avoid|no|not a fan of)\b.{0,20}\b(?:museums?|culture|nature|coast|beaches?)\b",
        message,
        re.I,
    ):
        return clarify()
    endpoint_clauses = re.findall(
        r"\bstart(?:ing)?(?:\s+(?:my|the|our)\s+trip)?\s+(?:from|in)\s+([^,.;!?]+)", message, re.I
    )
    if (
        any(
            not re.match(r"(?:los angeles|la|san francisco|sf)\b", value, re.I)
            or re.match(r"(?:los angeles|la|san francisco|sf)\s+(?:or\b|/)", value, re.I)
            for value in endpoint_clauses
        )
        or re.search(
            r"\b(?:end(?:ing)?|finish(?:ing)?|return(?:ing)?)\s+(?:in|at|to)\b|終點|终点|回到|返回", message, re.I
        )
        or any(
            not re.fullmatch(r"洛杉[磯矶]|舊金山|旧金山", value.strip())
            for value in re.findall(r"[從从]\s*([^，。！？,.;!?]+?)\s*出[發发]", message)
        )
    ):
        return clarify()
    if extension and (
        max(context.allowed_days) != 7
        or any(_number(m[1]) not in {3, 10} or re.fullmatch(r"週|周|星期|weeks?", m[2], re.I) for m in day_matches)
    ):
        if any(_number(m[1]) == 14 for m in day_matches):
            return clarify(
                "你是指總共 14 天、其中三天在優勝美地，還是 14 天再加三天（總共 17 天）？目前專用的優勝美地延長功能只支援 7 天延長至 10 天。尚未加入草稿或更改行程。"
                if chinese
                else "Do you mean 14 days total with three days at Yosemite, or 14 days plus three additional Yosemite days (17 total)? "
                "The dedicated Yosemite extension currently supports only a 7-day trip extended to 10 days. Nothing was staged or changed."
            )
        return clarify()
    if individualized and re.search(
        r"如果|假如|要是|\b(?:if|unless)\b|(?:不要|不想|不是)\s*(?:玩|旅行)?\s*(?:\d+|[一二兩两三四五六七八九十]+)\s*(?:天|週|周)|\b(?:not|never)\s+(?:\d+|ten|two)\s+(?:days?|weeks?)|\b(?:don't|do not)\s+want.{0,20}(?:days?|weeks?)|(?:不喜歡|不喜欢|不要).{0,8}(?:海岸|博物館|博物馆|自然)",
        message,
        re.I,
    ):
        return clarify()
    if re.search(
        r"預算|预算|budget|輪椅|轮椅|無障礙|无障碍|wheelchair|accessible|\d\s*月|\b(?:January|February|March|April|May|June|July|August|September|October|November|December|booked|reserved)\b|\b(?:depart|return)\s+on\b|\d{4}[-/]\d{1,2}[-/]\d{1,2}|每天.{0,10}(?:個|个)景點|每天.{0,10}(?:個|个)景点|\bno driving\b|已[預预][訂订]",
        message,
        re.I,
    ):
        question = (
            "目前可同時調整總天數、旅行節奏、興趣、起點、必去／排除景點和每日駕車上限。這條請求另含尚未建模的預算、日期或無障礙等條件，請先確認如何處理；沒有忽略條件生成草稿。"
            if chinese
            else "This request includes a budget, date, accessibility or other constraint this planner cannot yet enforce. Please clarify that condition; no partial draft was created."
        )
        return CopilotInterpretationV1(
            state="clarification_required",
            assistant_message=question,
            intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
        )
    if (
        individualized
        and yosemite
        and day_matches
        and not re.search(r"旅行|行程|旅遊|旅游|trip|itinerary|vacation", message, re.I)
    ):
        return clarify()
    if individualized and (day_matches and total_days is None or total_days is not None and not 1 <= total_days <= 21):
        question = (
            "請指定一個 1–21 天的總天數；例如十天或兩週。尚未更改行程。"
            if chinese
            else "Choose one total duration from 1 to 21 days. Nothing was changed."
        )
        return CopilotInterpretationV1(
            state="clarification_required",
            assistant_message=question,
            intents=(CopilotIntentV1(type="request_clarification", clarification_question=question),),
        )
    if re.search(
        r"(?:don['’]?t|do not|never)\s+(?:replan|change|modify|start|add|include|visit|go|avoid)|不要(?:重新|修改|更改|增加|加入)|不(?:要|想)(?:去|到)\s*(?:hearst|赫斯特|赫氏)|\b(?:book|pay|purchase)\b|預訂|预订|付款",
        message,
        re.I,
    ):
        return clarify()
    if (
        not (coastal or start or individualized or extension or official)
        and required_mentions
        and required_mentions <= set(context.allowed_stop_ids)
        and re.search(r"\b(?:before|after)\b", message, re.I)
    ):
        # Existing accepted-stop ordering has its own complete-request validator.
        return None
    time_text = message[: drive_match.start()] + message[drive_match.end() :] if drive_match else message
    if re.search(r"\b(?:minutes?|hours?|duration|time window|reorder)\b|分鐘|分钟|小時|小时|排序", time_text, re.I):
        return clarify()
    if official and big_sur and not re.search(r"修改|改[線线]|路[線线]|reroute|change|avoid|route", message, re.I):
        return None
    if islands and re.search(
        r"Channel Islands Harbor|Prisoners|Santa Rosa|Anacapa|San Miguel|camp|露[營营]|聖羅莎|圣罗莎"
        r"|(?:don['’]?t|do not|never|not)\s+(?:want\s+to\s+)?(?:go|visit|include|add)"
        r"|(?:不要|不想|別|别).{0,12}(?:Channel Islands|海[峽峡]群[島岛])",
        message,
        re.I,
    ):
        return clarify()
    if redwood and re.search(
        r"(?:don['’]?t|do not|never|not)\s+(?:want\s+to\s+)?(?:go|visit|include|add)"
        r"|(?:不要|不想|別|别).{0,12}(?:red\s*woods?|紅木|红木)",
        message,
        re.I,
    ):
        return clarify()
    if extension and re.search(
        r"不要|不想|取消|縮短|缩短|\b(?:not|never|cancel|book|pay|hours?)\b|預訂|预订|小時|小时", message, re.I
    ):
        return clarify()
    if re.search(rf"\bday\s+\d+\b|第\s*({number})\s*天", message, re.I):
        # Delegate only after checking the complete request's other constraints.
        # A fixed-day insertion must retain its exact road-evidence capability.
        if (
            not (coastal or start or individualized or extension or official)
            and re.search(r"\badd\b", message, re.I)
            and any(
                place["route_status"] == "registered_route_candidate"
                and any(str(alias).casefold() in message.casefold() for alias in (place["name"], *place["aliases"]))
                for place in context.known_places
            )
        ):
            return None
        return clarify(
            "目前整趟重排不能固定景點在指定日期。請確認是否可以讓系統選擇日期；尚未加入草稿。"
            if chinese
            else "Whole-trip replanning cannot pin a stop to a specified day. May the planner choose the day instead? Nothing was staged."
        )
    required, excluded = sorted(required_mentions), []
    if extension:
        required = sorted(set(required) | {"yosemite_national_park"})
    if redwood and "redwood_national_park" not in required:
        required.append("redwood_national_park")
    if islands and "channel_islands" not in required:
        required.append("channel_islands")
    if hearst and "hearst_castle" not in required:
        required.append("hearst_castle")
    for place_id, aliases in PLACE_ALIASES.items():
        for alias in aliases:
            if re.search(
                r"(?:不(?:是)?(?:很)?想去|避[開开]|不要去|略過|略过|skip|avoid|exclude|not keen on|instead of|rather than)\s*(?:the\s+)?"
                + re.escape(alias),
                message,
                re.I,
            ):
                excluded.append(place_id)
                break
    if set(required) & set(excluded):
        return clarify()
    if coastal and set(required) & {"yosemite_national_park", "sequoia_national_park", "kings_canyon_national_park"}:
        return clarify()
    city = None
    if start:
        value = next(v for v in start.groups() if v)
        city = "Los Angeles" if re.search(r"洛杉|^la$|los angeles", value, re.I) else "San Francisco"
    parameter_model = (
        YosemiteExtensionParameters
        if extension
        else IndividualizedTripParameters
        if individualized
        else TripPlanParameters
    )
    drive_limit = 720
    if drive_match:
        amount, unit = next((drive_match[i], drive_match[i + 1]) for i in (1, 3, 5) if drive_match[i])
        value = Decimal(amount) if re.fullmatch(r"\d+(?:\.\d+)?", amount) else Decimal(_number(amount))
        minutes = value * (60 if re.fullmatch(r"小時|小时|hours?", unit, re.I) else 1)
        if minutes != minutes.to_integral_value() or not 60 <= minutes <= 720:
            return clarify()
        drive_limit = int(minutes)
    parsed = parameter_model(
        **({"additional_yosemite_days": 3} if extension else {}),
        **(
            {
                "total_days": total_days or max(context.allowed_days),
                "pace": pace,
                "interests": interests,
                "profile_updates": (["pace"] if pace_specified else [])
                + (
                    ["interests"]
                    if interests or re.search(r"沒有特別偏好|没有特别偏好|no particular interests", message, re.I)
                    else []
                )
                + (["max_daily_drive_minutes"] if drive_match else [])
                + (["coastal"] if coastal else []),
            }
            if individualized
            else {}
        ),
        start_city=city,
        coastal=coastal,
        required_place_ids=required,
        excluded_place_ids=sorted(set(excluded)),
        avoid_regions=["big_sur"] if official and big_sur else [],
        verify_official=bool(official and big_sur or hearst or islands or redwood or extension),
        max_daily_drive_minutes=drive_limit,
    )
    descriptions = []
    if comparative_driving:
        descriptions.append(
            "檢查你指定的每日駕車上限；總駕車時間是否比原行程更少仍未驗證"
            if chinese
            else "check your explicit daily driving cap; less total driving than the original trip remains unverified"
        )
    if individualized:
        descriptions.append(
            f"重新規劃總共 {parsed.total_days} 天；本次指定節奏：{parsed.pace if pace_specified else '沿用已接受偏好（首次預設平衡）'}，每日開車上限：{str(drive_limit) + ' 分鐘' if drive_match else '沿用已接受上限（首次預設 720 分鐘）'}；保留既有必去景點，以道路可達的候選景點補充，興趣為 {', '.join(interests) or '沿用既有安排'}，不會自動增加三天優勝美地"
            if chinese
            else f"plan exactly {parsed.total_days} days, {pace + ' pace' if pace_specified else 'retaining the accepted pace (balanced on the first plan)'}, {('at most ' + str(drive_limit) + ' driving minutes per day') if drive_match else 'retaining the accepted daily driving limit (720 minutes on the first plan)'}; preserve existing visits and prefer {', '.join(interests) or 'the existing trip interests'}; no automatic Yosemite extension"
        )
    if extension:
        descriptions.append(
            "為優勝美地增加三天，把七天行程延長為十天；安排山谷瀑布與草甸、Glacier Point／Sentinel Dome、Mariposa Grove／Wawona 三組不同景點，附官方照片、簡介及可查證的遊客分享。共同進出道路可能重疊，住宿和季節交通仍需確認"
            if chinese
            else "add three distinct Yosemite day routes to extend seven days to ten, with official photos, introductions and attributed visitor reports when available; shared access roads may overlap and accommodation/seasonal access remain unconfirmed"
        )
    if city:
        descriptions.append(("把起點改為 " if chinese else "start from ") + city)
    if coastal:
        descriptions.append(
            "以海岸沿線景點為主，重新選擇過夜地點" if chinese else "prefer coastal stops and reselect overnight bases"
        )
    if excluded:
        descriptions.append(("回避 " if chinese else "exclude ") + ", ".join(PLACE_ALIASES[p][0] for p in excluded))
    ordinary_required = (
        set(required)
        - {"hearst_castle", "channel_islands", "redwood_national_park"}
        - ({"yosemite_national_park"} if extension else set())
    )
    if ordinary_required:
        descriptions.append(
            ("必去 " if chinese else "require ") + ", ".join(PLACE_ALIASES[p][0] for p in sorted(ordinary_required))
        )
    if hearst:
        descriptions.append(
            "把 Hearst Castle 加入必去約束，路線到訪客中心"
            if chinese
            else "require Hearst Castle via its visitor center"
        )
    if islands:
        descriptions.append(
            "加入 Channel Islands 一日遊：開車到 Ventura Harbor 的 Island Packers 港口，搭船往返 Santa Cruz Island / Scorpion Anchorage；地圖以紫色虛線標示海上示意路段，預留 480 分鐘，船班與船票尚未確認"
            if chinese
            else "require a Channel Islands day trip via Ventura Harbor and Santa Cruz Island / Scorpion Anchorage, with a purple dashed schematic sea route and a 480-minute allowance; sailing times and tickets remain unverified"
        )
    if redwood:
        descriptions.append(
            "把 Redwood National Park 加入必去約束，安排 Lady Bird Johnson Grove 森林步道，開車抵達 Bald Hills Road 步道停車場，預留 90 分鐘，並核對 NPS 道路與步道公告；允許以 Arcata 作為待選飯店的過夜區域"
            if chinese
            else "require Redwood National Park via the Lady Bird Johnson Grove trailhead on Bald Hills Road, allow 90 minutes for the forest walk, check NPS road and trail conditions, and allow an Arcata overnight area with the hotel still to be selected"
        )
    if official and big_sur:
        descriptions.append(
            "先核對州立公園與 Caltrans 官網，規劃避開 Big Sur 的替代路線"
            if chinese
            else "check State Parks and Caltrans, then evaluate a route avoiding Big Sur"
        )
    output = (
        (
            "我會"
            + "；".join(descriptions)
            + "。加入草稿後執行 Preview，才會取得路況、重新求解並檢查新方案；接受前原行程保持不變。"
        )
        if chinese
        else (
            "I can "
            + "; ".join(descriptions)
            + ". Add this request to the draft, then Preview to fetch evidence, solve and independently check the new plan. Acceptance remains separate."
        )
    )
    return CopilotInterpretationV2(
        state="proposal_ready",
        assistant_message=output,
        intents=(CopilotIntentV2(type=TRIP_REPLAN, parameters=parsed.model_dump()),),
    )
