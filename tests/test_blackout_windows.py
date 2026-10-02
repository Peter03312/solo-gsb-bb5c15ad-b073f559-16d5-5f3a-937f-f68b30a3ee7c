"""End-to-end tests for temporary spectrum control windows (blackouts)."""
from factories import make_payload


def submit(
    client,
    name="Event",
    freq=(100, 110),
    circle=(0, 0, 1000),
    start="2026-10-01T08:00:00Z",
    end="2026-10-01T12:00:00Z",
):
    r = client.post(
        "/api/applications",
        json=make_payload(name=name, freq=freq, circle=circle, start=start, end=end),
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def window_payload(
    freq=(105, 108),
    circle=(0, 0, 1000),
    start="2026-10-01T09:00:00Z",
    end="2026-10-01T11:00:00Z",
    name="VIP guard",
    reason="state visit",
):
    cx, cy, radius = circle
    low, high = freq
    return {
        "name": name,
        "circle": {"cx": cx, "cy": cy, "radius": radius},
        "frequency": {"low": low, "high": high},
        "time_window": {"start": start, "end": end},
        "reason": reason,
    }


def cur(client):
    return client.get("/api/revision/current").json()["current_revision"]


class TestValidation:
    def test_preview_rejects_illegal_geometry(self, client):
        r = client.post(
            "/api/blackout-windows/preview",
            json=window_payload(freq=(200, 100), circle=(0, 0, -2)),
        )
        assert r.status_code == 422
        codes = {d["code"] for d in r.json()["error"]["details"]}
        assert {
            "radius_not_positive",
            "frequency_low_above_high",
        } <= codes

    def test_enable_rejects_illegal_geometry_without_changing_revision(self, client):
        r = client.post(
            "/api/blackout-windows",
            json={**window_payload(freq=(200, 100)), "expected_revision": 0},
        )
        assert r.status_code == 422
        assert cur(client) == 0
        assert client.get("/api/blackout-windows").json()["windows"] == []

    def test_naive_time_rejected(self, client):
        payload = window_payload()
        payload["time_window"]["start"] = "2026-10-01T09:00:00"
        r = client.post("/api/blackout-windows/preview", json=payload)
        assert r.status_code == 422
        assert any(
            d["code"] == "naive_datetime"
            for d in r.json()["error"]["details"]
        )


class TestPreview:
    def test_preview_lists_hit_effective_and_waitlist_and_revision(self, client):
        a = submit(client, "A", freq=(100, 110))
        b = submit(client, "B", freq=(105, 108))
        c = submit(client, "C", freq=(200, 210))
        client.post(f"/api/applications/{a}/approve", json={"expected_revision": 0})
        client.post(f"/api/applications/{b}/waitlist", json={})
        # revision is now 1
        r = client.post(
            "/api/blackout-windows/preview", json=window_payload()
        )
        assert r.status_code == 200
        body = r.json()
        assert body["current_revision"] == 1
        assert body["hit_effective_application_ids"] == [a]
        assert body["hit_waitlist_application_ids"] == [b]
        assert body["hit_waitlist"][0]["queue_position"] == 1
        assert c not in body["hit_effective_application_ids"]
        # non-mutating
        assert cur(client) == 1
        assert client.get("/api/waitlist").json()["waitlist_application_ids"] == [b]

    def test_preview_endpoint_touch_and_time_touch_boundaries(self, client):
        # frequency endpoint touch hits; a pure gap does not
        a = submit(client, "A", freq=(100, 110))
        client.post(f"/api/applications/{a}/approve", json={"expected_revision": 0})
        touch = client.post(
            "/api/blackout-windows/preview",
            json=window_payload(freq=(110, 120)),
        ).json()
        assert touch["hit_effective_application_ids"] == [a]
        gap = client.post(
            "/api/blackout-windows/preview",
            json=window_payload(freq=(111, 120)),
        ).json()
        assert gap["hit_effective_application_ids"] == []
        # half-open time touch: window [12:00,13:00) vs license [08:00,12:00)
        time_touch = client.post(
            "/api/blackout-windows/preview",
            json=window_payload(
                freq=(100, 110),
                start="2026-10-01T12:00:00Z",
                end="2026-10-01T13:00:00Z",
            ),
        ).json()
        assert time_touch["hit_effective_application_ids"] == []

    def test_preview_guard_band_does_not_extend_window(self, client):
        # GUARD_BAND is 2 MHz in tests; a window band 1 MHz beside a license
        # still does NOT hit.
        a = submit(client, "A", freq=(100, 110))
        client.post(f"/api/applications/{a}/approve", json={"expected_revision": 0})
        body = client.post(
            "/api/blackout-windows/preview",
            json=window_payload(freq=(111, 112)),
        ).json()
        assert body["hit_effective_application_ids"] == []


class TestEnable:
    def test_stale_enable_is_refused_and_consumes_no_revision(self, client):
        submit(client, "A")
        client.post(
            "/api/blackout-windows",
            json={**window_payload(), "expected_revision": 0},
        )  # revision 1
        r = client.post(
            "/api/blackout-windows",
            json={**window_payload(freq=(200, 201)), "expected_revision": 0},
        )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "stale_revision"
        assert cur(client) == 1
        listing = client.get("/api/blackout-windows").json()
        assert listing["window_count"] == 1

    def test_enable_consumes_one_revision_and_reports_hits(self, client):
        a = submit(client, "A", freq=(100, 110))
        b = submit(client, "B", freq=(105, 108))
        client.post(f"/api/applications/{a}/approve", json={"expected_revision": 0})
        client.post(f"/api/applications/{b}/waitlist", json={})
        r = client.post(
            "/api/blackout-windows",
            json={**window_payload(), "expected_revision": 1},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["enabled"] is True
        assert body["revision"] == 2
        assert body["window_id"] == 1
        assert body["window"]["status"] == "active"
        assert body["window"]["enabled_revision"] == 2
        assert body["hit_effective_application_ids"] == [a]
        assert body["hit_waitlist_application_ids"] == [b]
        # enabling keeps the existing license effective and the queue intact
        snap = client.get("/api/revision/2").json()
        assert snap["effective_application_ids"] == [a]
        assert client.get("/api/waitlist").json()["waitlist_application_ids"] == [b]

    def test_enable_does_not_approve_waitlisted_applications(self, client):
        a = submit(client, "A", freq=(200, 210))
        b = submit(client, "B", freq=(200, 205))
        client.post(f"/api/applications/{a}/approve", json={"expected_revision": 0})
        client.post(f"/api/applications/{b}/waitlist", json={})
        # window far away from both still consumes exactly one revision and
        # does not drain anything
        r = client.post(
            "/api/blackout-windows",
            json={
                **window_payload(freq=(900, 910)),
                "expected_revision": 1,
            },
        )
        assert r.status_code == 201
        assert r.json()["revision"] == 2
        assert client.get("/api/waitlist").json()["waitlist_application_ids"] == [b]


class TestPreviewAndApprovalGating:
    def _setup(self, client):
        a = submit(client, "A", freq=(100, 110))
        b = submit(client, "B", freq=(105, 108))
        client.post(f"/api/applications/{a}/approve", json={"expected_revision": 0})
        client.post(
            "/api/blackout-windows",
            json={**window_payload(), "expected_revision": 1},
        )  # revision 2
        return a, b

    def test_pending_preview_reports_window(self, client):
        _, b = self._setup(client)
        body = client.post(f"/api/applications/{b}/preview").json()
        assert body["has_window_hits"] is True
        assert body["hit_blackout_window_ids"] == [1]
        assert body["window_hits"][0]["window"]["id"] == 1
        assert body["window_hits"][0]["trigger"] == {
            "time_intersect": True,
            "area_intersect": True,
            "frequency_intersect": True,
        }
        # preview never mutates
        assert cur(client) == 2

    def test_hit_approval_refused_no_revision_consumed(self, client):
        _, b = self._setup(client)
        r = client.post(
            f"/api/applications/{b}/approve", json={"expected_revision": 2}
        )
        assert r.status_code == 409
        err = r.json()["error"]
        assert err["code"] == "blackout_window_hit"
        app_detail = err["details"]["applications"][0]
        assert app_detail["application_id"] == b
        assert app_detail["window_hits"][0]["window"]["id"] == 1
        # B also interferes with A; that conflict is reported on the same
        # member for a single actionable response.
        assert app_detail["conflicts"][0]["application"]["id"] == 1
        assert cur(client) == 2
        assert client.get(f"/api/applications/{b}").json()["status"] == "pending"

    def test_non_hit_application_still_approved(self, client):
        self._setup(client)
        d = submit(client, "D", freq=(300, 310))
        r = client.post(
            f"/api/applications/{d}/approve", json={"expected_revision": 2}
        )
        assert r.status_code == 200
        assert r.json()["revision"] == 3

    def test_batch_hit_rejects_whole_group_and_reports_each_member(self, client):
        _, b = self._setup(client)
        d = submit(client, "D", freq=(300, 310))
        r = client.post(
            "/api/applications/batch-approve",
            json={"expected_revision": 2, "application_ids": [b, d]},
        )
        assert r.status_code == 409
        err = r.json()["error"]
        assert err["code"] == "blackout_window_hit"
        ids = [a["application_id"] for a in err["details"]["applications"]]
        assert ids == [b]
        assert cur(client) == 2
        assert client.get(f"/api/applications/{d}").json()["status"] == "pending"

    def test_available_ranges_exclude_window_positions(self, client):
        _, _ = self._setup(client)
        # A [100,110] is still effective: guard band 2 forbids (95,112)
        # open for a width-3 candidate; the window [105,108] additionally
        # forbids the closed region [102,108], which fuses into (95,112).
        x = submit(client, "X", freq=(100, 103))
        body = client.post(
            f"/api/applications/{x}/available-ranges",
            json={"candidate_low_min": 90, "candidate_low_max": 130},
        ).json()
        ranges = [
            (r["min"], r["max"], r["min_inclusive"], r["max_inclusive"])
            for r in body["available_ranges"]
        ]
        assert ranges == [(90, 95, True, True), (112, 130, True, True)]
        assert body["active_blackout_window_ids"] == [1]
        assert cur(client) == 2

    def test_available_ranges_window_alone_makes_closed_endpoints(self, client):
        # No effective licenses: only the window closed region applies.
        client.post(
            "/api/blackout-windows",
            json={**window_payload(freq=(105, 108)), "expected_revision": 0},
        )
        x = submit(client, "X", freq=(100, 103))
        body = client.post(
            f"/api/applications/{x}/available-ranges",
            json={"candidate_low_min": 90, "candidate_low_max": 130},
        ).json()
        ranges = [
            (r["min"], r["max"], r["min_inclusive"], r["max_inclusive"])
            for r in body["available_ranges"]
        ]
        assert ranges == [(90, 102, True, False), (108, 130, False, True)]


class TestResumeGating:
    def test_resume_hitting_window_is_refused_and_license_stays_suspended(self, client):
        a = submit(client, "A", freq=(100, 110))
        client.post(f"/api/applications/{a}/approve", json={"expected_revision": 0})
        # suspend A (revision 2)
        client.post(
            f"/api/applications/{a}/suspend",
            json={"expected_revision": 1, "reason": "downtime"},
        )
        # enable a window covering A's band (revision 3)
        client.post(
            "/api/blackout-windows",
            json={**window_payload(), "expected_revision": 2},
        )
        r = client.post(
            f"/api/applications/{a}/resume", json={"expected_revision": 3}
        )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "blackout_window_hit"
        assert cur(client) == 3
        assert client.get(f"/api/applications/{a}").json()["status"] == "suspended"

        # lifting the window lets the resume proceed
        wid = client.get(
            "/api/blackout-windows", params={"status": "lifted"}
        ).json()
        lifted = client.post(
            "/api/blackout-windows/1/lift", json={"expected_revision": 3}
        )
        assert lifted.status_code == 200
        r = client.post(
            f"/api/applications/{a}/resume", json={"expected_revision": 4}
        )
        assert r.status_code == 200
        assert r.json()["original_approval_revision"] == 1


class TestWaitlistDrainWithWindow:
    def test_blocked_standby_keeps_queue_and_drain_continues(self, client):
        a = submit(client, "A", freq=(100, 110))
        b = submit(client, "B", freq=(105, 108))  # will hit the window
        d = submit(client, "D", freq=(300, 310))  # clean after release
        client.post(f"/api/applications/{a}/approve", json={"expected_revision": 0})
        client.post(f"/api/applications/{b}/waitlist", json={})
        client.post(f"/api/applications/{d}/waitlist", json={})
        # window covering B only
        client.post(
            "/api/blackout-windows",
            json={
                **window_payload(freq=(105, 108)),
                "expected_revision": 1,
            },
        )  # rev 2
        # revoke A (rev 3): B is blocked by the window and keeps its place,
        # D passes and auto-approves (rev 4).
        r = client.post(f"/api/applications/{a}/revoke", json={"reason": "x"})
        body = r.json()
        assert body["revision"] == 3
        assert body["final_revision"] == 4
        assert body["auto_approved_application_ids"] == [d]
        assert body["remaining_waitlist_application_ids"] == [b]
        pending = body["waitlist_pending"][0]
        assert pending["application_id"] == b
        assert pending["window_hits"][0]["window"]["id"] == 1
        assert cur(client) == 4

        # lifting the window (rev 5) does NOT auto-approve B
        client.post(
            "/api/blackout-windows/1/lift", json={"expected_revision": 4}
        )
        assert client.get("/api/waitlist").json()["waitlist_application_ids"] == [b]

        # the next release (revoke D, rev 6) lets B auto-approve (rev 7)
        r = client.post(f"/api/applications/{d}/revoke", json={"reason": "y"})
        assert r.json()["auto_approved_application_ids"] == [b]
        assert r.json()["final_revision"] == 7
        assert client.get("/api/waitlist").json()["waitlist_application_ids"] == []


class TestLift:
    def test_lift_arbitration_and_terminal_rules(self, client):
        submit(client, "A")
        client.post(
            "/api/blackout-windows",
            json={**window_payload(), "expected_revision": 0},
        )  # rev 1
        stale = client.post(
            "/api/blackout-windows/1/lift", json={"expected_revision": 0}
        )
        assert stale.status_code == 409
        assert stale.json()["error"]["code"] == "stale_revision"
        assert cur(client) == 1

        unknown = client.post(
            "/api/blackout-windows/999/lift", json={"expected_revision": 1}
        )
        assert unknown.status_code == 404

        ok = client.post(
            "/api/blackout-windows/1/lift",
            json={"expected_revision": 1, "reason": "all clear"},
        )
        assert ok.status_code == 200
        assert ok.json()["revision"] == 2
        assert ok.json()["window"]["status"] == "lifted"
        assert ok.json()["window"]["lifted_revision"] == 2

        again = client.post(
            "/api/blackout-windows/1/lift", json={"expected_revision": 2}
        )
        assert again.status_code == 409
        assert again.json()["error"]["code"] == "blackout_window_not_active"
        assert cur(client) == 2

    def test_lift_only_affects_later_decisions(self, client):
        a = submit(client, "A", freq=(100, 110))
        b = submit(client, "B", freq=(105, 108))
        client.post(f"/api/applications/{a}/approve", json={"expected_revision": 0})
        client.post(
            "/api/blackout-windows",
            json={**window_payload(), "expected_revision": 1},
        )  # rev 2
        blocked = client.post(
            f"/api/applications/{b}/approve", json={"expected_revision": 2}
        )
        assert blocked.status_code == 409
        client.post(
            "/api/blackout-windows/1/lift", json={"expected_revision": 2}
        )  # rev 3
        # B is still pending; now it only faces the ordinary effective set
        # (A), i.e. ordinary interference, no window field.
        prev = client.post(f"/api/applications/{b}/preview").json()
        assert prev["hit_blackout_window_ids"] == []
        assert prev["current_revision"] == 3

    def test_list_and_filter_windows(self, client):
        client.post(
            "/api/blackout-windows",
            json={**window_payload(), "expected_revision": 0},
        )
        listing = client.get("/api/blackout-windows").json()
        assert listing["window_count"] == 1
        one = client.get("/api/blackout-windows/1").json()["window"]
        assert one["enabled_revision"] == 1
        assert client.get("/api/blackout-windows/999").status_code == 404
        bad_filter = client.get("/api/blackout-windows", params={"status": "nope"})
        assert bad_filter.status_code == 422


class TestHistory:
    def test_snapshots_and_comparison_restore_windows(self, client):
        a = submit(client, "A", freq=(100, 110))
        client.post(f"/api/applications/{a}/approve", json={"expected_revision": 0})
        # rev 2: enable
        client.post(
            "/api/blackout-windows",
            json={**window_payload(), "expected_revision": 1},
        )
        before = client.get("/api/revision/1").json()
        assert before["active_blackout_window_ids"] == []
        assert before["blackout_window_events"] == []

        at2 = client.get("/api/revision/2").json()
        assert at2["active_blackout_window_ids"] == [1]
        assert at2["active_blackout_windows"][0]["frequency"] == {
            "low": 105.0,
            "high": 108.0,
        }
        event = at2["blackout_window_events"][0]
        assert event["revision"] == 2 and event["kind"] == "enabled"
        assert event["window"]["id"] == 1
        # original approval event is untouched
        assert at2["approvals"][0]["revision"] == 1
        assert "window" not in at2["approvals"][0]

        # rev 3: lift
        client.post(
            "/api/blackout-windows/1/lift", json={"expected_revision": 2}
        )
        at3 = client.get("/api/revision/3").json()
        assert at3["active_blackout_window_ids"] == []
        kinds = [(e["revision"], e["kind"]) for e in at3["blackout_window_events"]]
        assert kinds == [(2, "enabled"), (3, "lifted")]

        cmp_ = client.get(
            "/api/revision/compare",
            params={"base_revision": 1, "target_revision": 3},
        ).json()
        window_sets = cmp_["active_blackout_window_ids"]
        for key in ("base", "target", "added", "removed"):
            assert window_sets[key] == []
        types = [(e["revision"], e["type"]) for e in cmp_["blackout_window_events"]]
        assert types == [
            (2, "blackout_window_enable"),
            (3, "blackout_window_lift"),
        ]
        assert all(
            e["type"] in {"blackout_window_enable", "blackout_window_lift"}
            for e in cmp_["differences"]
            if "window_id" in e
        )

        # current snapshot exposes active windows and the event log
        current = client.get("/api/revision/current").json()
        assert current["active_blackout_window_ids"] == []
        assert len(current["blackout_window_events"]) == 2

    def test_window_active_inside_compare_interval(self, client):
        client.post(
            "/api/blackout-windows",
            json={**window_payload(), "expected_revision": 0},
        )  # rev 1
        cmp_ = client.get(
            "/api/revision/compare",
            params={"base_revision": 0, "target_revision": 1},
        ).json()
        assert cmp_["active_blackout_window_ids"]["added"] == [1]
        assert cmp_["active_blackout_window_ids"]["base"] == []
        assert cmp_["active_blackout_window_ids"]["target"] == [1]
        assert cmp_["active_blackout_window_ids"]["removed"] == []
