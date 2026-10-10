import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from workflow_fixture import prepared_dataset, evaluation_run
import training_workflow as wf


class FakeYOLO:
    # Intentionally has NO validator attribute: matches Model.val's contract.
    names={0:'smoke',1:'fire'}
    emit_callback=True

    def __init__(self, path):
        self.callback=None

    def add_callback(self,event,callback):
        if event != 'on_val_end':
            raise AssertionError(event)
        self.callback=callback

    def val(self, **args):
        # Actual output is deliberately different from the requested name.
        out=Path(args['project'])/(args['name']+'_actual')
        out.mkdir(parents=True)
        (out/'nested').mkdir()
        (out/'nested/plot.png').write_bytes(b'plot')
        if self.emit_callback:
            self.callback(SimpleNamespace(save_dir=out))
        box=SimpleNamespace(ap_class_index=[0,1], class_result=lambda i:(.6,.7,.5,.4))
        return SimpleNamespace(box=box,results_dict={'metrics/mAP50(B)':.5},speed={'inference':1.2})


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.data=prepared_dataset(self.root)
        self.run=evaluation_run(self.root,self.data)

    def evaluate_fake(self, **kwargs):
        with patch.dict(sys.modules, {'ultralytics':SimpleNamespace(YOLO=FakeYOLO)}):
            return wf.evaluate(self.run,self.data,self.root/'local',**kwargs)

    def test_callback_directory_and_summary_are_saved(self):
        result=self.evaluate_fake()
        self.assertEqual([c['name'] for c in result['per_class']],['smoke','fire'])
        self.assertEqual(len(list((self.run/'evaluations').glob('*/nested/plot.png'))),1)
        self.assertEqual(wf.read_json(self.run/'status.json')['state'],'validated')
        self.assertEqual(wf.collect_results(self.root/'drive')[0]['metrics/mAP50(B)'],.5)

    def test_missing_callback_fails_instead_of_guessing_output(self):
        with patch.object(FakeYOLO,'emit_callback',False):
            with self.assertRaisesRegex(RuntimeError,'output directory'):
                self.evaluate_fake()

    def test_current_label_tampering_rejected_before_inference(self):
        (self.data/'val/labels/c.txt').write_text('0 0.3 0.3 0.2 0.2\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Prepared label changed'):
            self.evaluate_fake()

    def test_yaml_redirect_rejected(self):
        path=self.data/'fire_v001_colab.yaml'
        path.write_text(path.read_text().replace('val: val/images','val: test/images'),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'YAML val path'):
            self.evaluate_fake()

    def test_wrong_model_classes_rejected(self):
        with patch.object(FakeYOLO,'names',{0:'person',1:'car'}):
            with self.assertRaisesRegex(ValueError,'trained D-Fire'):
                self.evaluate_fake()

    def test_test_eval_keeps_validation_summary_and_status(self):
        self.evaluate_fake()
        before=(self.run/'validation_summary.json').read_bytes()
        result=self.evaluate_fake(split='test',test_approved=True)
        self.assertEqual(result['split'],'test')
        self.assertEqual((self.run/'validation_summary.json').read_bytes(),before)
        self.assertEqual(wf.read_json(self.run/'status.json')['state'],'validated')

    def test_extra_image_not_listed_in_manifest_rejected(self):
        (self.data/'val/images/extra.jpg').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError,'Image directory differs'):
            self.evaluate_fake()


if __name__=='__main__':
    unittest.main()
