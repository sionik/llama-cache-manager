from datetime import datetime

import pytest

from llama_cache_manager.age import CutoffError, describe, parse_cutoff

NOW = datetime(2026, 8, 21, 12, 0, 0)


class TestParseCutoff:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("7d", datetime(2026, 8, 14, 12, 0, 0)),
            ("7 days", datetime(2026, 8, 14, 12, 0, 0)),
            ("7 days ago", datetime(2026, 8, 14, 12, 0, 0)),
            ("12h", datetime(2026, 8, 21, 0, 0, 0)),
            ("1w", datetime(2026, 8, 14, 12, 0, 0)),
            ("30 minutes", datetime(2026, 8, 21, 11, 30, 0)),
        ],
    )
    def test_counts_a_span_back_from_now(self, spec, expected):
        assert parse_cutoff(spec, NOW) == expected

    def test_reads_a_date(self):
        assert parse_cutoff("2026-07-01", NOW) == datetime(2026, 7, 1)

    def test_reads_a_date_and_time(self):
        assert parse_cutoff("2026-07-01 18:30", NOW) == datetime(2026, 7, 1, 18, 30)

    def test_names_the_unit_it_could_not_read(self):
        with pytest.raises(CutoffError, match="fortnights"):
            parse_cutoff("3 fortnights", NOW)

    @pytest.mark.parametrize("spec", ["", "   ", "soon", "-3d", "3", "d7"])
    def test_rejects_text_that_is_not_a_cutoff(self, spec):
        with pytest.raises(CutoffError):
            parse_cutoff(spec, NOW)


class TestDescribe:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "just now"),
            (59, "just now"),
            (60, "1 minute ago"),
            (3600, "1 hour ago"),
            (86400, "1 day ago"),
            (86400 * 27, "27 days ago"),
            (86400 * 45, "1 month ago"),
            (86400 * 400, "1 year ago"),
        ],
    )
    def test_reads_as_one_unit(self, seconds, expected):
        assert describe(seconds) == expected

    def test_treats_a_future_stamp_as_now(self):
        assert describe(-500) == "just now"


class TestCutoffsBeyondTheCalendar:
    @pytest.mark.parametrize("spec", ["100000y", "999999999999d", "0001-01-01"])
    def test_reports_a_cutoff_the_calendar_cannot_hold(self, spec):
        from llama_cache_manager.age import cutoff

        with pytest.raises(CutoffError, match="beyond the calendar"):
            cutoff(spec, NOW)

    def test_gives_a_timestamp_for_a_cutoff_it_can_hold(self):
        from llama_cache_manager.age import cutoff

        assert cutoff("7d", NOW) == datetime(2026, 8, 14, 12, 0, 0).timestamp()
