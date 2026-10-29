"""Bounded reads of official travel sources; page content is evidence only."""

from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from html.parser import HTMLParser

from .island_excursion import CHANNEL_URLS
from .workspace import WorkspaceError
from .yosemite_extension import YOSEMITE_URLS

REDWOOD_URLS = {
    "redwood_lodging": "https://www.nps.gov/redw/planyourvisit/lodging.htm",
    "redwood_walk": "https://www.nps.gov/thingstodo/lbjgrove.htm",
    "redwood_conditions": "https://www.nps.gov/redw/planyourvisit/conditions.htm",
}
OFFICIAL_URLS = {
    **REDWOOD_URLS,
    **YOSEMITE_URLS,
    **CHANNEL_URLS,
    "big_sur_park": "https://www.parks.ca.gov/?page_id=570",
    "highway_1": "https://roads.dot.ca.gov/roadscell.php?roadnumber=1",
    "hearst_castle": "https://hearstcastle.org/plan/faqs/",
}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip += 1
        if tag in {"p", "div", "h1", "h2", "h3", "li", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise WorkspaceError("official_source_redirect_blocked", 409)


def read_official_sources(*, big_sur: bool, hearst: bool, channel_islands: bool = False, redwood: bool = False, yosemite: bool = False) -> list[dict]:
    names = (["big_sur_park", "highway_1"] if big_sur else []) + (["hearst_castle"] if hearst else [])
    if channel_islands:
        names += list(CHANNEL_URLS)
    if redwood:
        names += list(REDWOOD_URLS)
    if yosemite:
        names += list(YOSEMITE_URLS)
    result = []
    opener = urllib.request.build_opener(_NoRedirect)
    for name in names:
        url = OFFICIAL_URLS[name]
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ItineraryRepair/0.3", "Accept": "text/html"})
            with opener.open(request, timeout=8) as response:
                if response.status != 200 or response.geturl() != url:
                    raise WorkspaceError("official_source_response_invalid", 409)
                raw = response.read(1_000_001)
                if len(raw) > 1_000_000:
                    raise WorkspaceError("official_source_response_too_large", 409)
        except (OSError, urllib.error.URLError, TimeoutError):
            raise WorkspaceError("official_source_unavailable", 409) from None
        parser = _Text()
        parser.feed(raw.decode("utf-8", errors="replace"))
        lines = [re.sub(r"\s+", " ", line).strip() for line in "".join(parser.parts).splitlines()]
        lines = [line for line in lines if line]
        selected = [
            line
            for line in lines
            if re.search(
                r"clos|reopen|fire|alert|restriction|one.way|1.way|checked|latest reported|bus|shuttle|tour|arriv|reserv|boat|Ventura|Scorpion|Bald Hills|Lady Bird|old.growth|lodging|Arcata",
                line,
                re.I,
            )
        ]
        expected = ({name:[r"Yosemite|Sentinel Dome"]} if name in YOSEMITE_URLS else {
            "redwood_lodging": [r"Staying Near Redwood National and State Parks"],
            "redwood_walk": [r"Walk the Lady Bird Johnson Grove"],
            "redwood_conditions": [r"Redwood National and State Parks"],
            "channel_islands_nps": [r"Island Packers", r"Ventura Harbor"],
            "channel_islands_operator": [r"Santa Cruz Island", r"Scorpion Anchorage"],
            "big_sur_park": [r"Pfeiffer Big Sur"],
            "highway_1": [r"SR 1", r"STATE ROUTE 1"],
            "hearst_castle": [r"Hearst Castle"],
        })[name]
        if (name not in YOSEMITE_URLS and len(selected) < 2) or not any(re.search(term, " ".join(lines), re.I) for term in expected):
            raise WorkspaceError("official_source_content_unrecognized", 409)
        plain = " ".join(lines)
        if name in YOSEMITE_URLS:
            evidence_text = " ".join(line for line in lines if re.search(r"Glacier Point Road|firefighter|Mariposa|shuttle|trail|Wawona|Meadow|Sentinel|open|clos",line,re.I))
        elif name == "redwood_lodging":
            if "Arcata Chamber of Commerce" not in plain:
                raise WorkspaceError("official_source_content_unrecognized", 409)
            evidence_text = plain[plain.index("Staying Near Redwood National and State Parks") :]
        elif name == "redwood_walk":
            if not all(term in plain for term in ("Lady Bird Johnson", "Bald Hills", "45-60 Minutes")):
                raise WorkspaceError("official_source_content_unrecognized", 409)
            evidence_text = plain[plain.rindex("Walk the Lady Bird Johnson Grove") :]
        elif name == "redwood_conditions":
            if not all(term in plain for term in ("Bald Hills Road", "Trails", "park trails not listed below are open")):
                raise WorkspaceError("official_source_content_unrecognized", 409)
            evidence_text = plain[plain.index("Roads & Day Use Areas") :]
        elif name == "channel_islands_nps":
            if not all(term in plain for term in ("Island Packers", "Ventura Harbor", "1691 Spinnaker")):
                raise WorkspaceError("official_source_content_unrecognized", 409)
            evidence_text = plain[plain.index("Public Boat Transportation") :]
        elif name == "channel_islands_operator":
            if not all(term in plain for term in ("From Ventura Harbor", "Scorpion Anchorage", "45 minutes")):
                raise WorkspaceError("official_source_content_unrecognized", 409)
            evidence_text = " ".join(line for line in lines if re.search(r"Ventura|Scorpion|Travel time|45 minutes", line, re.I))
        elif name == "highway_1":
            central = re.search(r"\[IN THE CENTRAL CALIFORNIA AREA\](.*?)(?:\[IN THE NORTHERN|$)", plain, re.I)
            if central is None:
                raise WorkspaceError("official_source_content_unrecognized", 409)
            as_of = re.search(r"This highway information.*?(?:PM|AM)\.", plain, re.I)
            evidence_text = (as_of.group(0) if as_of else "") + " Central California SR 1: " + central.group(1)
        elif name == "big_sur_park":
            evidence_text = plain[plain.index("Current Restrictions") :]
        else:
            evidence_text = plain[plain.index("Are tour reservations required") :]
        excerpt = " ".join(evidence_text.split()[:170])
        park_access = None
        if name == "big_sur_park":
            says_closed = bool(re.search(r"(?:park|SP) is currently Closed|park is currently Closed", plain, re.I))
            says_reopen = bool(
                re.search(r"(?:will reopen|reopen(?:ed)? on|reopening).*?October 16,? 2026", plain, re.I)
            )
            says_open = bool(re.search(r"(?:park|SP) is currently Open", plain, re.I))
            park_access = (
                "conflicting"
                if says_closed and (says_reopen or says_open)
                else "closed"
                if says_closed
                else "open"
                if says_open
                else "unknown"
            )
        if name == "redwood_conditions":
            road = re.search(r"Bald Hills Road\s+(Open(?:/\s*Hazard Conditions)?|Closed)\b", plain, re.I)
            trail_closed = re.search(r"Lady Bird Johnson(?: Grove)?(?: Trail)?\s+(?:Closed|Partially Open)", plain, re.I)
            park_access = "closed" if trail_closed or road and road.group(1).lower() == "closed" else (
                "open_with_hazards" if road and "hazard" in road.group(1).lower() else "open" if road else "unknown"
            )
        if name == "yosemite_conditions":
            section = re.search(r"Glacier Point Road(.*?)Mariposa Grove Road",plain,re.I)
            park_access = "open_with_hazards" if section and re.search(r"\bOpen\s*$",section.group(1)) else "closed_or_unknown"
        result.append(
            {
                "source_id": name,
                "url": url,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "body_sha256": hashlib.sha256(raw).hexdigest(),
                "excerpt": excerpt,
                "source_dates": sorted(
                    set(
                        re.findall(
                            r"\b\d{1,2}/\d{1,2}/\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}",
                            " ".join(lines),
                        )
                    )
                )[:12],
                "scope": "Current official bulletin at retrieval time; no future trip-date assurance",
                "status": "retrieved",
                "park_access": park_access,
                "big_sur_highway_closure": bool(
                    name == "highway_1"
                    and re.search(r"is closed", evidence_text, re.I)
                    and re.search(r"Monterey|Big Sur", evidence_text, re.I)
                ),
                "private_vehicle_to_castle": False if name == "hearst_castle" else None,
            }
        )
    return result
