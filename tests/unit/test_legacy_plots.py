"""Exercise retained plotting entry points with sparse synthetic records."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from edge_llm_lab.visualization.unreferenced_plots import UnreferencedPlots


def test_patient_plot_accepts_missing_latency_without_fabricating_zero(tmp_path, monkeypatch):
    observed = []
    original = plt.close
    def inspect_and_close(figure=None):
        if hasattr(figure, 'axes'):
            observed.extend(p.get_height() for p in figure.axes[2].patches)
        original(figure)
    monkeypatch.setattr(plt, 'close', inspect_and_close)
    record = {'patient_name': 'synthetic', 'patient_type': 'successful',
              'test_result': 'TIMEOUT', 'status': 'INCOMPLETE', 'rounds_count': 1,
              'total_duration_sec': 1, 'collected_data': {}, 'expected_data': {},
              'rounds_data': [{'round': 1, 'status': 'INCOMPLETE',
                               'latency_breakdown': {'total_ms': None}}]}
    UnreferencedPlots().plot_per_patient_summary(record, str(tmp_path), 'test')
    assert (tmp_path / 'patient_synthetic_test.png').is_file()
    assert all(value != 0 for value in observed)
    assert not plt.get_fignums()
