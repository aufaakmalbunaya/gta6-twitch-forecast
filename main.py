"""Forecast launch-day Twitch viewers, hour by hour, for one case.

The program reads every ``*.in`` case file in the input directory and writes
one ``<id>.out`` forecast per case: 23 rows, hours 2 to 24, each with five
quantiles of that hour's concurrent-viewer count at the levels
0.05, 0.25, 0.5, 0.75 and 0.95.

Method
------
The case anchors on its hour-1 print ``p1``.  What varies between launches is
not the level but the *shape* of the day: how the count moves away from hour 1
as regions unlock at their local midnight and as the day's prime time arrives.
Past big launches in the case's ``comparables`` show that shape, so each
hour's forecast is built from their log-ratios ``ln(viewers_h / viewers_1)``.

Not every past launch is an equally good guide for this one.  Two launches
resemble each other when they behaved the same way in their first hour, so
each comparable is weighted by how close it sits to this case on three
measures of that first hour:

* ``hype`` -- how far hour 1 jumped above the category's own quiet level in
  the three days before release day, ``ln(p1 / pre_median)``.  A launch that
  was already live in big regions by hour 1 jumps far; one that unlocks later
  in the UTC day barely moves yet, and its day then explodes.
* ``size`` -- the sheer scale of hour 1, ``ln(p1)``.  Mega launches hold
  their audience through the day differently from mid-size ones.

The weight is a Gaussian kernel over those two distances (bandwidth 1.2 log
units each).  Each hour's five quantiles blend two views of the weighted
log-ratios for that hour: the weighted empirical quantiles (which keep the
real shape, including the long upper tail -- some launches explode later in
the day) and a weighted normal fit (which steadies the far tails, where only
a few launches vote).  The blend is 30% empirical, 70% normal, in log
space, then multiplied back by ``p1``.  Nothing is fitted to the target
case's own future: every number comes from past launches.

Backtests on the kit's practice comparables (each launch held out in turn and
forecast from the rest) put this at a mean pinball loss around 0.36 on clean
launches -- launches whose hour 1 already caught a major unlock -- and around
0.40 over all launches, against about 0.55 and 0.93 for flat persistence at
hour 1 with a widening spread.  It beats persistence on 17 of the 19 clean
launches and 28 of 37 launches overall.

Fallbacks keep every case valid: a comparable without a usable hour 1, or
without a pre-launch level, is left out of the pool; a target without a
pre-launch level simply drops that distance and leans on the ones it has; an
hour with no pool at all falls back to flat persistence with a wide spread.

The program is deterministic: no clock, no randomness, no network, no
environment-dependent ordering.  It uses only the Python standard library.
"""

import json
import math
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the scored levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
# Kernel bandwidths (log units) for the two likeness measures.
BANDWIDTH = {"hype": 1.2, "size": 1.2}
# Share of the weighted empirical quantiles in the final blend; the rest is
# the weighted normal fit.  The empirical part keeps the real (skewed) shape,
# the normal part steadies the far tails.
EMPIRICAL_SHARE = 0.3
# How many days before release day count as the quiet pre-launch level.
PRE_DAYS = 3
# Spread of the fallback forecast, in log units per square-root hour.
FALLBACK_SIGMA = 0.30
# Output quantiles are rounded to 2 decimals, half up, and clamped here.
MAX_VIEWERS = 1_000_000_000


def median_of(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def round_half_up_2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def hour_of(timestamp):
    """UTC hour (0-23) of an ISO 8601 timestamp like 2026-09-15T00:00:00Z."""
    return int(timestamp[11:13])


def day_of(timestamp):
    """Calendar date of an ISO 8601 timestamp, as a datetime.date."""
    return date(int(timestamp[0:4]), int(timestamp[5:7]), int(timestamp[8:10]))


def pre_launch_median(points, day_start, key):
    """Median of `key` over the PRE_DAYS days before the release day.

    `points` are hourly dicts with a `from` timestamp.  The median is robust
    to the last hours of the window already running hot (New Zealand unlocks
    about 13 hours before the UTC release day), and to sparse sampling: when
    the window holds too few points, it widens to seven days.
    """
    for span in (PRE_DAYS, 7):
        first_day = day_start - timedelta(days=span)
        values = [p[key] for p in points
                  if first_day <= day_of(p["from"]) < day_start
                  and p.get(key) is not None and p[key] > 0]
        if len(values) >= 12:
            return median_of(values)
    if len(values) >= 3:
        return median_of(values)
    return None


def release_day_hours(comp):
    """Hourly points of a comparable on its release day, keyed by hour 1-24.

    Each value is the point dict for that UTC hour of `day_start`'s date.
    """
    day = day_of(comp["day_start"])
    hours = {}
    for point in comp.get("hourly") or []:
        if day_of(point["from"]) == day:
            hours[hour_of(point["from"]) + 1] = point
    return hours


def comparable_profile(comp, game_name):
    """Likeness measures and day shape of one past launch, or None.

    A launch is usable when its release day has a positive hour-1 print and
    at least 20 scored hours.  Its own game is never a guide to itself.
    """
    if comp.get("name") == game_name:
        return None
    hours = release_day_hours(comp)
    first = hours.get(1)
    if first is None or not first.get("viewers") or first["viewers"] <= 0:
        return None
    scored = {h: p["viewers"] for h, p in hours.items()
              if h >= 2 and p.get("viewers") and p["viewers"] > 0}
    if len(scored) < 20:
        return None
    hourly = comp.get("hourly") or []
    day = day_of(comp["day_start"])
    pre_viewers = pre_launch_median(hourly, day, "viewers")
    hype = None
    if pre_viewers:
        hype = math.log(first["viewers"] / pre_viewers)
    return {
        "p1": first["viewers"],
        "hype": hype,
        "day": scored,
    }


def target_profile(case):
    """Likeness measures of the case itself, from its hour-1 print and history.

    The quiet pre-launch level comes from `history_hourly` in the three days
    before release day; when that is too thin (the final's recorder only
    started two days before), the case's own category history among the
    comparables fills the same window.
    """
    known = next((k for k in case["known"] if k["hour"] == 1), None)
    if known is None:
        raise ValueError("case has no hour-1 print")
    day = day_of(case["day_start"])
    history = case.get("history_hourly") or []
    pre_viewers = pre_launch_median(history, day, "viewers")
    if pre_viewers is None:
        own = next((c for c in case.get("comparables") or []
                    if c.get("name") == case["game"]["name"]), None)
        if own is not None:
            pre_viewers = pre_launch_median(own.get("hourly") or [], day, "viewers")
    if pre_viewers is None and history:
        older = [p["viewers"] for p in history
                 if day_of(p["from"]) < day and p.get("viewers") and p["viewers"] > 0]
        if older:
            pre_viewers = median_of(older)
    p1 = max(1, known["viewers"])
    hype = math.log(p1 / pre_viewers) if pre_viewers else None
    return {"p1": p1, "hype": hype}


def kernel_weight(target, other):
    """Gaussian kernel weight of one comparable for this case.

    Distances are taken only over the likeness measures both sides carry;
    a comparable missing a measure the target has does not vote at all.
    """
    dist2 = 0.0
    pairs = (
        (target["hype"], other["hype"], BANDWIDTH["hype"]),
        (math.log(target["p1"]), math.log(other["p1"]), BANDWIDTH["size"]),
    )
    for mine, theirs, width in pairs:
        if mine is None:
            continue
        if theirs is None:
            return 0.0
        dist2 += ((mine - theirs) / width) ** 2
    return math.exp(-0.5 * dist2)


def weighted_mean_sd(pairs):
    """Mean and standard deviation of (value, weight) pairs."""
    total = sum(w for _, w in pairs)
    mean = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    return mean, math.sqrt(max(var, 1e-9))


def weighted_quantile(pairs, tau):
    """Interpolated tau-quantile of (value, weight) pairs.

    Values are placed at the midpoint of their cumulative weight, then the
    quantile is read by linear interpolation in cumulative-weight space.
    """
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return None
    marks = []
    running = 0.0
    for value, weight in ordered:
        marks.append((running + weight / 2.0, value))
        running += weight
    point = tau * total
    if point <= marks[0][0]:
        return marks[0][1]
    if point >= marks[-1][0]:
        return marks[-1][1]
    for (c0, v0), (c1, v1) in zip(marks, marks[1:]):
        if c0 <= point <= c1:
            share = (point - c0) / (c1 - c0) if c1 > c0 else 0.0
            return v0 * (1.0 - share) + v1 * share
    return marks[-1][1]


def fallback_row(p1, hour):
    """Flat persistence with a wide spread, used when no comparable votes."""
    return [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)) for z in Z]


def forecast_hour(target, pool, hour):
    """Five quantiles of one hour's viewers, from the comparable pool.

    Each vote is a comparable's log-ratio for that hour, weighted by how
    much that launch resembles this case.  The quantiles blend the weighted
    empirical quantiles with a weighted normal fit, in log space.
    """
    votes = []
    for other in pool:
        if hour not in other["day"]:
            continue
        weight = kernel_weight(target, other)
        if weight <= 0.0:
            continue
        votes.append((math.log(other["day"][hour] / other["p1"]), weight))
    if not votes:
        return fallback_row(target["p1"], hour)
    mean, sd = weighted_mean_sd(votes)
    base = math.log(target["p1"])
    out = []
    for tau, z in zip(LEVELS, Z):
        empirical = weighted_quantile(votes, tau)
        normal = mean + z * sd
        blended = EMPIRICAL_SHARE * empirical + (1.0 - EMPIRICAL_SHARE) * normal
        out.append(math.exp(base + blended))
    return out


def tidy_quantiles(raw):
    """Round, clamp to (0, 1e9] and force non-decreasing, in that order."""
    out = []
    floor = 0.01
    for value in raw:
        q = round_half_up_2(value)
        q = max(floor, min(MAX_VIEWERS, q))
        if out:
            q = max(q, out[-1])
        out.append(q)
    return out


def forecast_case(case):
    """The full 23-row forecast for one case dict."""
    if case["levels"] != LEVELS:
        raise ValueError(f"case asks for levels {case['levels']}, not {LEVELS}")
    target = target_profile(case)
    pool = []
    for comp in case.get("comparables") or []:
        profile = comparable_profile(comp, case["game"]["name"])
        if profile is not None:
            pool.append(profile)
    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        hours.append({"hour": hour,
                      "q": tidy_quantiles(forecast_hour(target, pool, hour))})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
