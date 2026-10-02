"""Unit tests for temporary spectrum control window (blackout) rules."""
from datetime import datetime, timedelta, timezone

from app.adjudication import (
    BlackoutWindow,
    Circle,
    License,
    available_ranges,
    evaluate_windows,
    window_forbidden_lower_bounds,
    window_hits_license,
)

T0 = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
T1 = T0 + timedelta(hours=2)
T2 = T1
T3 = T0 + timedelta(hours=4)


def lic(
    app_id,
    cx=0.0,
    cy=0.0,
    r=10.0,
    flo=100.0,
    fhi=110.0,
    start=T0,
    end=T1,
):
    return License(
        application_id=app_id,
        circle=Circle(cx, cy, r),
        freq_low=flo,
        freq_high=fhi,
        start_time=start,
        end_time=end,
    )


def win(
    wid,
    cx=0.0,
    cy=0.0,
    r=10.0,
    flo=100.0,
    fhi=110.0,
    start=T0,
    end=T1,
):
    return BlackoutWindow(
        window_id=wid,
        circle=Circle(cx, cy, r),
        freq_low=flo,
        freq_high=fhi,
        start_time=start,
        end_time=end,
        enabled_revision=5,
    )


class TestWindowHits:
    def test_full_overlap_hits(self):
        hits = evaluate_windows(lic(1), [win(7)])
        assert [w.window_id for w in hits] == [7]
        assert window_hits_license(win(7), lic(1))

    def test_frequency_endpoint_touch_counts(self):
        # license [100,110] vs window [110,120]: closed bands touch at 110
        assert window_hits_license(win(7, flo=110, fhi=120), lic(1))
        assert window_hits_license(win(7, flo=80, fhi=100), lic(1))

    def test_guard_band_gap_does_not_extend_window(self):
        # a 1 MHz gap beside the window is NOT a hit regardless of any
        # guard band (windows never carry a guard band)
        assert not window_hits_license(win(7, flo=111, fhi=112), lic(1))
        assert not evaluate_windows(lic(1), [win(7, flo=111, fhi=112)])

    def test_circles_must_intersect(self):
        assert not window_hits_license(
            win(7, cx=100.0, r=1.0), lic(1, r=1.0)
        )

    def test_circle_tangent_counts(self):
        # centre distance 10 == radii 5+5
        assert window_hits_license(
            win(7, cx=10.0, r=5.0), lic(1, cx=0.0, r=5.0)
        )

    def test_half_open_time_touch_does_not_hit(self):
        # window starts exactly when license ends -> no time intersection
        assert not window_hits_license(
            win(7, start=T2, end=T3), lic(1, start=T0, end=T1)
        )

    def test_partial_time_overlap_hits(self):
        assert window_hits_license(
            win(7, start=T0 + timedelta(hours=1), end=T3), lic(1)
        )

    def test_multiple_windows_returned_in_order(self):
        windows = [
            win(7, flo=200, fhi=210),  # miss
            win(8, flo=105, fhi=108),  # hit
            win(9, flo=100, fhi=100),  # hit (single frequency inside)
        ]
        assert [w.window_id for w in evaluate_windows(lic(1), windows)] == [
            8,
            9,
        ]

    def test_single_frequency_band_touch(self):
        assert window_hits_license(
            win(7, flo=110, fhi=110), lic(1, flo=110, fhi=120)
        )


class TestWindowForbiddenLowerBounds:
    def test_closed_interval(self):
        # candidate width 5, window [110,120] -> [105,120]
        bounds = window_forbidden_lower_bounds(
            lic(1, flo=100, fhi=105), win(7, flo=110, fhi=120)
        )
        assert bounds == (105.0, 120.0)

    def test_disjoint_time_or_area_returns_none(self):
        assert (
            window_forbidden_lower_bounds(
                lic(1, start=T0, end=T1), win(7, start=T2, end=T3)
            )
            is None
        )
        assert (
            window_forbidden_lower_bounds(
                lic(1, cx=100.0, r=1.0), win(7, r=1.0)
            )
            is None
        )


def tuples(ranges):
    return [(r.min, r.max, r.min_inclusive, r.max_inclusive) for r in ranges]


class TestAvailableRangesWithWindows:
    def test_window_removes_closed_region(self):
        # width 5, window [110,120] forbids [105,120] (closed endpoints)
        ranges = available_ranges(
            lic(1, flo=100, fhi=105), [], 0.0, 90, 130, [win(7, flo=110, fhi=120)]
        )
        assert tuples(ranges) == [
            (90, 105, True, False),
            (120, 130, False, True),
        ]

    def test_window_ignores_guard_band(self):
        # Even with guard band 2, the window region stays closed: a gap of 1
        # beside the window band is forbidden by the window (endpoint touch
        # rule on the shifted band), and a position at the window boundary
        # stays forbidden.
        ranges = available_ranges(
            lic(1, flo=100, fhi=105),
            [],
            2.0,
            90,
            130,
            [win(7, flo=110, fhi=120)],
        )
        assert tuples(ranges) == [
            (90, 105, True, False),
            (120, 130, False, True),
        ]

    def test_window_and_open_guard_interval_meet(self):
        # guard open (103,122); window closed [110,120] (candidate width 5,
        # window band [115,120]).  Union leaves 103 attainable, 122
        # attainable, middle closed.
        ranges = available_ranges(
            lic(1, flo=100, fhi=105),
            [lic(9, flo=110, fhi=120)],
            2.0,
            90,
            130,
            [win(7, flo=115, fhi=120)],
        )
        assert tuples(ranges) == [
            (90, 103, True, True),
            (122, 130, True, True),
        ]

    def test_closed_window_touching_open_guard_interval_fuses(self):
        # guard band 0 would give closed [105,120]; here guard 2 from a
        # different grant plus window: window [110,114] -> [105,114] closed,
        # grant [120,130] -> (113,132) open.  They overlap -> fused, left
        # endpoint 105 forbidden (window), right endpoint 132 attainable.
        ranges = available_ranges(
            lic(1, flo=100, fhi=105),
            [lic(9, flo=120, fhi=130)],
            2.0,
            90,
            140,
            [win(7, flo=110, fhi=114)],
        )
        assert tuples(ranges) == [
            (90, 105, True, False),
            (132, 140, True, True),
        ]

    def test_disjoint_area_window_does_not_shrink_ranges(self):
        ranges = available_ranges(
            lic(1, flo=100, fhi=105),
            [],
            0.0,
            90,
            130,
            [win(7, flo=110, fhi=120, cx=1000.0, r=1.0)],
        )
        assert tuples(ranges) == [(90, 130, True, True)]

    def test_reported_points_agree_with_window_verdicts(self):
        grants = [lic(9, flo=200, fhi=205)]
        windows = [win(7, flo=110, fhi=120)]
        candidate = lic(1, flo=0, fhi=5)
        ranges = available_ranges(candidate, grants, 2.0, 90, 230, windows)
        points = {90, 230, 105, 120}
        for rr in ranges:
            points.update({rr.min, rr.max})
        for x in points:
            shifted = lic(1, flo=x, fhi=x + 5)
            free = not evaluate_windows(shifted, windows)
            attainable = any(
                rr.min <= x <= rr.max
                and (x != rr.min or rr.min_inclusive)
                and (x != rr.max or rr.max_inclusive)
                for rr in ranges
            )
            assert free == attainable, (x, ranges)
