from ep_pressure_pvt_fluids.plots.shoe_tests import norm_type, scatter_tests, legend_handles


def test_types_are_normalised():
    out = norm_type(["LOT", "fit", " XLOT ", "XLO", None, "LOT / FIT", "weird"]).tolist()
    assert out == ["LOT", "FIT", "XLOT", "XLOT", "UNDIFFERENTIATED", "UNDIFFERENTIATED", "UNDIFFERENTIATED"]


def test_every_type_has_its_own_marker():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    counts = scatter_tests(ax, [1, 2, 3, 4], [1, 2, 3, 4], ["LOT", "FIT", "XLOT", None])
    assert counts == {"LOT": 1, "FIT": 1, "XLOT": 1, "UNDIFFERENTIATED": 1}
    assert len({h.get_marker() for h in legend_handles()}) == 4
    plt.close(fig)
