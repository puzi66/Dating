from matching.simulate import format_report, gini, main, run_simulation


def test_gini_extremes():
    assert gini([1, 1, 1, 1]) == 0.0
    assert gini([0, 0, 0, 4]) == 0.75
    assert gini([]) == 0.0


def test_small_simulation_runs_and_respects_invariants():
    res = run_simulation(n_users=200, n_weeks=3, seed=5, learn=True)
    assert len(res.weeks) == 3
    for w in res.weeks:
        assert w.max_load <= 3
        assert w.coverage_feasible >= 0.98
        assert w.gini_bmatch < w.gini_naive
        assert 0 <= w.second_date_rate <= 1
    report = format_report(res, None, 200, 5)
    assert "Gini" in report and "second-date" in report.lower()


def test_cli_smoke(capsys):
    main(["--users", "120", "--weeks", "2", "--seed", "2"])
    out = capsys.readouterr().out
    assert "cover" in out and "naive" in out
