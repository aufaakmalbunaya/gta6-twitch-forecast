"""Forecast launch-day Twitch viewers, hour by hour, for one case.

Round 2. The program reads every ``*.in`` case file in the input directory
and writes one ``<id>.out`` forecast per case: 23 rows, hours 2 to 24, each
with five quantiles of that hour's concurrent-viewer count at the levels
0.05, 0.25, 0.5, 0.75 and 0.95.

Method
------
The case anchors on its hour-1 print ``p1``. What varies between launches is
not the level but the *shape* of the day: how the count moves away from hour 1
as regions unlock at their local midnight and as the day's prime time arrives.
Past big launches in the case's ``comparables`` show that shape, so each
hour's forecast is built from their log-ratios ``ln(viewers_h / viewers_1)``.

The shape depends most on sheer scale: the bigger hour 1 already is, the
less room the day has left to grow. Across the kit's past launches the
correlation of a launch's median log-ratio with ``ln(p1)`` is about -0.82,
and with its peak log-ratio about -0.94. A mega launch's day decays from an
already huge hour 1 and recovers toward it in the evening; a small launch's
day explodes a hundredfold as the world wakes up. So the median log-ratio
for each hour blends two views of that scale dependence:

* a local-linear fit in ``ln(p1)`` (Gaussian kernel, bandwidth 2.0 log
  units): a line through the pool's log-ratios, read at this case's own
  scale. Unlike a weighted average it extrapolates past the biggest past
  launch, which is where a record launch like this one sits;
* the round-1 kernel estimate: a Gaussian kernel over likeness in hype
  (``ln(p1 / pre_median)``) and size (bandwidth 1.2 each), blending the
  weighted empirical median (30%) with the weighted mean (70%).

The blend is 70% local-linear, 30% kernel. Hype on its own adds nothing
once scale is fitted -- it is nearly the same signal (correlation 0.91 with
the pre-midnight ramp) -- and extra features only add variance, so the
median uses scale alone.

The spread is calibrated, not assumed. Every past launch is forecast in
turn from the rest of the pool (leave-one-out), and the misses are kept as
residuals. For this case each hour's spread is the standard deviation of
those residuals, weighted toward launches of a similar scale (Gaussian
kernel on ``ln(p1)``, bandwidth 1.5). A big launch's spread comes from big
launches, whose days are predictable; a small launch's spread comes from
small launches, whose days are not. The five quantiles are the median plus
the standard normal quantiles times that spread, in log space, then
multiplied back by ``p1``.

Backtests on the kit's practice comparables (each launch held out in turn
and forecast from the rest, 37 launches) put this at a mean pinball loss
around 0.37 overall and 0.14 on the biggest launches, against 0.40 and 0.16
for the round-1 entry and 0.93 for flat persistence; 90% coverage is 0.90,
where round 1 covered 0.87 on the same backtest. Nothing is fitted to the
target case's own future: every number comes from past launches.

Fallbacks keep every case valid: a comparable without a usable hour 1, or
without a pre-launch level, is left out of the pool; a target without a
pre-launch level simply drops that distance and leans on the ones it has;
an hour with no pool at all falls back to flat persistence with a wide
spread; an hour with too few residuals falls back to the pool's own spread.

The program is deterministic: no clock, no randomness, no network, no
environment-dependent ordering. It uses only the Python standard library.
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
# Kernel bandwidth (log units) of the local-linear fit in ln(p1).
LOCLIN_BANDWIDTH = 2.0
# Kernel bandwidths (log units) for the round-1 likeness measures.
KERNEL_BANDWIDTH = {"hype": 1.2, "size": 1.2}
# Share of the local-linear median in the final median; the rest is the
# kernel estimate. The local-linear part extrapolates in scale; the kernel
# part keeps the estimate anchored to the most similar past launches.
LOCLIN_SHARE = 0.7
# Share of the weighted empirical median in the kernel estimate; the rest
# is the weighted mean.
EMPIRICAL_SHARE = 0.3
# Kernel bandwidth (log units) weighting residuals by scale similarity.
RESIDUAL_BANDWIDTH = 1.5
# How many days before release day count as the quiet pre-launch level.
PRE_DAYS = 3
# Spread of the fallback forecast, in log units per square-root hour.
FALLBACK_SIGMA = 0.30
# Floor on the calibrated spread, in log units: with almost no residual
# variation the quantiles would collapse onto the median.
MIN_SPREAD = 0.05
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
    The profile carries the log-ratios for each scored hour, so the median
    and spread fits never re-read the raw points.
    """
    if comp.get("name") == game_name:
        return None
    hours = release_day_hours(comp)
    first = hours.get(1)
    if first is None or not first.get("viewers") or first["viewers"] <= 0:
        return None
    p1 = first["viewers"]
    log_ratios = {h: math.log(p["viewers"] / p1) for h, p in hours.items()
                  if h >= 2 and p.get("viewers") and p["viewers"] > 0}
    if len(log_ratios) < 20:
        return None
    hourly = comp.get("hourly") or []
    day = day_of(comp["day_start"])
    pre_viewers = pre_launch_median(hourly, day, "viewers")
    hype = None
    if pre_viewers:
        hype = math.log(first["viewers"] / pre_viewers)
    return {
        "p1": p1,
        "log_p1": math.log(p1),
        "hype": hype,
        "log_ratios": log_ratios,
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
    return {"p1": p1, "log_p1": math.log(p1), "hype": hype}


def kernel_weight(target_log_p1, target_hype, other, bandwidth):
    """Gaussian kernel weight of one comparable for this case.

    Distances are taken only over the likeness measures both sides carry;
    a comparable missing a measure the target has does not vote at all.
    """
    dist2 = ((other["log_p1"] - target_log_p1) / bandwidth) ** 2
    if target_hype is not None:
        if other["hype"] is None:
            return 0.0
        dist2 += ((other["hype"] - target_hype)
                  / KERNEL_BANDWIDTH["hype"]) ** 2
    return math.exp(-0.5 * dist2)


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


def kernel_median(target, pool, hour):
    """Round-1 kernel estimate of one hour's median log-ratio.

    Gaussian kernel over hype and size; the median blends the weighted
    empirical median (30%) with the weighted mean (70%).
    """
    votes = []
    for other in pool:
        if hour not in other["log_ratios"]:
            continue
        weight = kernel_weight(target["log_p1"], target["hype"], other,
                               KERNEL_BANDWIDTH["size"])
        if weight > 0.0:
            votes.append((other["log_ratios"][hour], weight))
    if not votes:
        return None
    total = sum(w for _, w in votes)
    mean = sum(v * w for v, w in votes) / total
    empirical = weighted_quantile(votes, 0.5)
    return EMPIRICAL_SHARE * empirical + (1.0 - EMPIRICAL_SHARE) * mean


def local_linear_median(target, pool, hour):
    """Local-linear fit of one hour's median log-ratio against ln(p1).

    A weighted least-squares line through the pool's log-ratios, centred
    on this case's own scale, so the intercept is the prediction.  The line
    extrapolates past the biggest past launch, where a record launch sits.
    The slope is shrunk by a small ridge toward the weighted mean.
    """
    sum_w = 0.0
    sum_wd = 0.0
    sum_wd2 = 0.0
    sum_wy = 0.0
    sum_wdy = 0.0
    for other in pool:
        ratio = other["log_ratios"].get(hour)
        if ratio is None:
            continue
        d = other["log_p1"] - target["log_p1"]
        weight = math.exp(-0.5 * (d / LOCLIN_BANDWIDTH) ** 2)
        sum_w += weight
        sum_wd += weight * d
        sum_wd2 += weight * d * d
        sum_wy += weight * ratio
        sum_wdy += weight * d * ratio
    if sum_w <= 0.0:
        return None
    # Normal equations for [intercept, slope], ridge on the slope only.
    a11 = sum_w
    a12 = sum_wd
    a22 = sum_wd2 + 1e-4
    b1 = sum_wy
    b2 = sum_wdy
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        return b1 / a11
    return (b1 * a22 - b2 * a12) / det


def blend_median(target, pool, hour):
    """The hour's median log-ratio: 70% local-linear, 30% kernel."""
    loclin = local_linear_median(target, pool, hour)
    kernel = kernel_median(target, pool, hour)
    if loclin is None:
        return kernel
    if kernel is None:
        return loclin
    return LOCLIN_SHARE * loclin + (1.0 - LOCLIN_SHARE) * kernel


def leave_one_out_residuals(pool):
    """Leave-one-out misses of the blended median, per hour.

    Each past launch is forecast from the rest of the pool, and the miss
    (actual log-ratio minus forecast median) is kept.  These residuals are
    what the spread is calibrated from, so no launch ever calibrates its
    own forecast.
    """
    residuals = {hour: [] for hour in range(2, 25)}
    for index, held in enumerate(pool):
        others = pool[:index] + pool[index + 1:]
        target = {"log_p1": held["log_p1"], "hype": held["hype"]}
        for hour, actual in held["log_ratios"].items():
            forecast = blend_median(target, others, hour)
            if forecast is not None:
                residuals[hour].append((held["log_p1"], actual - forecast))
    return residuals


def calibrated_spread(target, residuals, hour):
    """Spread of one hour's log-ratio, from similar-scale residuals.

    The standard deviation of the leave-one-out residuals for that hour,
    weighted toward launches of a similar scale to this case.  Returns
    None when too few residuals vote.
    """
    votes = []
    for log_p1, residual in residuals.get(hour, []):
        d = (log_p1 - target["log_p1"]) / RESIDUAL_BANDWIDTH
        votes.append((residual, math.exp(-0.5 * d * d)))
    total = sum(w for _, w in votes)
    if total <= 0.0 or len(votes) < 3:
        return None
    mean = sum(r * w for r, w in votes) / total
    var = sum(w * (r - mean) ** 2 for r, w in votes) / total
    return max(math.sqrt(max(var, 0.0)), MIN_SPREAD)


def pool_spread(target, pool, hour):
    """Fallback spread: the pool's own weighted spread at that hour."""
    votes = []
    for other in pool:
        ratio = other["log_ratios"].get(hour)
        if ratio is None:
            continue
        weight = kernel_weight(target["log_p1"], target["hype"], other,
                               KERNEL_BANDWIDTH["size"])
        if weight > 0.0:
            votes.append((ratio, weight))
    if not votes:
        return None
    total = sum(w for _, w in votes)
    mean = sum(v * w for v, w in votes) / total
    var = sum(w * (v - mean) ** 2 for v, w in votes) / total
    return max(math.sqrt(max(var, 1e-9)), MIN_SPREAD)


def fallback_row(p1, hour):
    """Flat persistence with a wide spread, used when no comparable votes."""
    return [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)) for z in Z]


def forecast_hour(target, pool, residuals, hour):
    """Five quantiles of one hour's viewers, from the comparable pool."""
    median = blend_median(target, pool, hour)
    if median is None:
        return fallback_row(target["p1"], hour)
    spread = calibrated_spread(target, residuals, hour)
    if spread is None:
        spread = pool_spread(target, pool, hour)
    if spread is None:
        return fallback_row(target["p1"], hour)
    base = math.log(target["p1"])
    return [math.exp(base + median + z * spread) for z in Z]


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
    residuals = leave_one_out_residuals(pool) if pool else {}
    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        hours.append({"hour": hour,
                      "q": tidy_quantiles(forecast_hour(target, pool,
                                                        residuals, hour))})
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
