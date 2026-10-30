"""Summary tables agree with the stored results."""
from src import report


def test_scenario_counts():
    rows = {r["set"]: r for r in report.scenario_stats()}
    assert (rows["Held-out"]["scenarios"], rows["Held-out"]["user_turns"]) == (70, 267)
    assert (rows["Held-out 2"]["scenarios"], rows["Held-out 2"]["user_turns"]) == (60, 226)
    assert rows["Held-out"]["report_infeasible"] == 19 and rows["Held-out"]["clarify"] == 12


def test_full_agent_results():
    agent = {r["system"]: r for r in report.systems()}["Full agent"]
    assert report.fmt(agent["test_llm_satisfaction"]) == "0.990" and agent["test_llm_relaxations"] == 3
    assert report.fmt(agent["test2_llm_satisfaction"]) == "0.988" and agent["test2_llm_relaxations"] == 3


def test_no_relaxation_with_a_correct_state():
    for row in report.policy():
        assert row["relaxations_state_correct"] == 0


def test_error_and_claim_totals():
    total = report.extraction_errors()[-1]
    assert (total["rules"], total["llm"]) == (44, 23)
    held_out = {r["set"]: r for r in report.claims()}["Held-out"]
    assert held_out["claims"] == 27491 and held_out["unsupported"] == 0


def test_rounding_half_up():
    assert report.fmt(0.7525) == "0.753"
