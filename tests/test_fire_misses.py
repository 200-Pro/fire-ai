import unittest
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import workflow_fixture  # adds project scripts to sys.path
from analyze_fire_misses import analyze, box_iou, match_fire, missed_box_hint


def pred(box=(0, 0, 10, 10), score=.9, cls=1):
    return dict(xyxy=list(box), score=score, class_id=cls)


class FireMissTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(box_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1/3)

    def test_smoke_cannot_match_fire(self):
        r = match_fire([[0, 0, 10, 10]], [pred(cls=0)], .25)
        self.assertEqual((r['tp'], r['fn'], r['fp']), (0, 1, 0))

    def test_one_prediction_cannot_cover_two_truth_boxes(self):
        r = match_fire([[0, 0, 10, 10]]*2, [pred()], .25)
        self.assertEqual((r['tp'], r['fn'], r['fp']), (1, 1, 0))

    def test_duplicate_predictions_count_as_fp(self):
        r = match_fire([[0, 0, 10, 10]], [pred(), pred(score=.7)], .25)
        self.assertEqual((r['tp'], r['fn'], r['fp']), (1, 0, 1))

    def test_lower_confidence_can_recover_low_score_candidate(self):
        truth, predictions = [[0, 0, 10, 10]], [pred(score=.15)]
        self.assertEqual(match_fire(truth, predictions, .25)['fn'], 1)
        self.assertEqual(match_fire(truth, predictions, .10)['fn'], 0)
        self.assertEqual(missed_box_hint(truth[0], predictions)['hint'], 'low_confidence_candidate')

    def test_background_fp_and_no_predictions(self):
        self.assertEqual(match_fire([], [pred()], .25)['fp'], 1)
        self.assertEqual(match_fire([[0, 0, 10, 10]], [], .25)['fn'], 1)

    def test_hint_not_causal_claim(self):
        box = [0, 0, 10, 10]
        self.assertEqual(missed_box_hint(box, [pred(cls=0)])['hint'], 'smoke_overlap_review')
        self.assertEqual(missed_box_hint(box, [])['hint'], 'no_overlapping_fire_at_conf005')

    def test_complete_report_and_examples_without_changing_model_or_status(self):
        import numpy as np

        class FakeYOLO:
            names = {0: 'smoke', 1: 'fire'}

            def __init__(self, path):
                pass

            def predict(self, **kwargs):
                self_args = kwargs
                assert self_args['conf'] == .05 and self_args['save'] is False
                for name, score in [('c.jpg', .9), ('d.jpg', .15)]:
                    yield SimpleNamespace(path=str(Path(kwargs['source']) / name), orig_shape=(64, 64),
                        boxes=[SimpleNamespace(cls=np.array(1), conf=np.array(score),
                                               xyxy=np.array([[19.2, 19.2, 44.8, 44.8]]))])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = workflow_fixture.prepared_dataset(root)
            run = workflow_fixture.evaluation_run(root, data)
            weights = (run / 'final/best.pt').read_bytes()
            status = (run / 'status.json').read_bytes()
            with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=FakeYOLO)}):
                result = analyze(run, data, root / 'analysis', device='cpu', max_examples=1)
            summary = json.loads((result / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['images'], 2)
            self.assertEqual(summary['baseline_missed_boxes'], 1)
            self.assertEqual(summary['hint_counts'], {'low_confidence_candidate': 1})
            low = next(r for r in summary['threshold_sweep'] if r['confidence'] == .1)
            high = next(r for r in summary['threshold_sweep'] if r['confidence'] == .25)
            self.assertEqual((low['tp'], low['fp'], low['fn']), (1, 1, 0))
            self.assertEqual((high['tp'], high['fp'], high['fn']), (0, 1, 1))
            self.assertEqual(len(list((result / 'examples').glob('*.jpg'))), 1)
            self.assertFalse((result / 'weights').exists())
            self.assertEqual((run / 'final/best.pt').read_bytes(), weights)
            self.assertEqual((run / 'status.json').read_bytes(), status)

    def test_notebook_json_and_python_cells_are_valid(self):
        path = Path(__file__).resolve().parents[1] / 'notebooks/dfire_fire_miss_analysis.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(notebook['nbformat'], 4)
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                # The user may save real Colab execution output back to Drive.
                self.assertTrue(cell['execution_count'] is None or isinstance(cell['execution_count'], int))
                self.assertIsInstance(cell['outputs'], list)
                compile(''.join(cell['source']), f'cell-{index}', 'exec')


if __name__ == '__main__':
    unittest.main()
