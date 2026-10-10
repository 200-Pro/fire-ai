import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import training_workflow as wf
from prepare_dfire import normalize_label_text


class BackupTests(unittest.TestCase):
    def test_failed_backup_keeps_previous_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local, remote = root / 'local', root / 'remote'
            local.mkdir(); remote.mkdir()
            for name in ('last.pt', 'best.pt', 'results.csv'):
                (local / name).write_bytes(b'checkpoint')
            trainer = SimpleNamespace(epoch=0, last=local/'last.pt', best=local/'best.pt', save_dir=local)
            wf.save_checkpoint(trainer, remote)
            before = (remote/'latest_checkpoint.json').read_bytes()
            trainer.epoch = 1
            with patch.object(wf, 'copy_verified', side_effect=IOError('Drive disconnected')):
                with self.assertRaises(IOError):
                    wf.save_checkpoint(trainer, remote)
            self.assertEqual(before, (remote/'latest_checkpoint.json').read_bytes())
            self.assertFalse((remote/'checkpoints/epoch_0002/manifest.json').exists())

    def test_restore_checks_hashes_and_keeps_historical_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local, remote, data = root/'local', root/'remote', root/'data'
            local.mkdir(); remote.mkdir(); data.mkdir()
            for name in ('last.pt', 'best.pt', 'results.csv'):
                (local/name).write_bytes(name.encode())
            for folder in (remote, data):
                (folder/'split_v001.csv').write_text('fixed split', encoding='utf-8')
            summary = {'source_sha256': {'dataset':'a','splits':'b'}, 'fold':1,
                       'classes':{'0':'smoke','1':'fire'}, 'yaml': str(data/'data.yaml'),
                       'label_policy':'v2', 'manifest_sha256':'m', 'corrections_sha256':'c'}
            wf.write_json(remote/'dataset_summary.json', summary)
            trainer = SimpleNamespace(epoch=3,last=local/'last.pt',best=local/'best.pt',save_dir=local)
            wf.save_checkpoint(trainer, remote)
            target = root/'restored'
            wf.restore_checkpoint(remote, target, summary)
            self.assertEqual((target/'weights/best.pt').read_bytes(), b'best.pt')
            (remote/'checkpoints/epoch_0004/last.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Corrupt'):
                wf.restore_checkpoint(remote, root/'bad', summary)

    def test_training_and_test_are_opt_in(self):
        with self.assertRaises(ValueError):
            wf.train(None,None,None,None,'member')
        with self.assertRaises(ValueError):
            wf.evaluate(None,None,None,split='test')

    def test_run_ids_unique_and_path_safe(self):
        self.assertNotEqual(wf.make_id('member2'), wf.make_id('member2'))
        for bad in ('../x','a/b','','a b'):
            with self.assertRaises(ValueError):
                wf.make_id(bad)


class LabelTests(unittest.TestCase):
    def test_zero_area_clip_duplicates_and_idempotence(self):
        text = '0 0.5 0.5 0 0\n0 0.5 0.5 1.1 1\n1 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n'
        cleaned, fixes = normalize_label_text(text, 'sample.txt')
        self.assertEqual([x['action'] for x in fixes], ['drop_zero_area','clip_to_image','drop_duplicate'])
        self.assertIn('0 0.5 0.5 1 1', cleaned)
        self.assertEqual(normalize_label_text(cleaned, 'sample.txt'), (cleaned, []))

    def test_invalid_class_and_nonfinite_are_not_silently_repaired(self):
        for line in ('2 0.5 0.5 0.2 0.2', '0 nan 0.5 0.2 0.2', '0 0.5 0.5 -1 1'):
            with self.assertRaises(ValueError):
                normalize_label_text(line, 'bad.txt')


class NotebookTests(unittest.TestCase):
    def test_setup_evicts_old_workflow_and_checks_version(self):
        notebook = json.loads((SCRIPTS.parent/'notebooks/dfire_team_training.ipynb').read_text(encoding='utf-8'))
        setup = ''.join(next(c for c in notebook['cells'] if c['cell_type']=='code' and 'import uuid\n' in c['source'])['source'])
        self.assertLess(setup.index('sys.modules.pop'), setup.index('from training_workflow import'))
        self.assertIn(wf.WORKFLOW_VERSION, setup)

    def test_notebook_parses_and_dangerous_switches_default_false(self):
        notebook = json.loads((SCRIPTS.parent/'notebooks/dfire_team_training.ipynb').read_text(encoding='utf-8'))
        import ast
        found = set()
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(''.join(cell['source']), filename=f'cell {index}')
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in ('RUN_FULL_TRAIN','RUN_TEST_EVAL','FINAL_RUN_SELECTED','LABELS_REVIEWED'):
                            self.assertIsInstance(node.value, ast.Constant)
                            self.assertIs(node.value.value, False)
                            found.add(target.id)
        self.assertEqual(len(found), 4)


class OrchestrationTests(unittest.TestCase):
    def test_training_handoff_with_fake_trainer_no_gpu_or_training(self):
        class FakeYOLO:
            def __init__(self, model):
                self.callback = None
            def add_callback(self, event, callback):
                self.callback = callback
            def train(self, **args):
                path = Path(args['save_dir'])
                (path/'weights').mkdir(parents=True)
                (path/'weights/last.pt').write_bytes(b'optimizer checkpoint')
                (path/'weights/best.pt').write_bytes(b'best weights')
                (path/'results.csv').write_text('epoch,metric\n1,0.5\n', encoding='utf-8')
                self.trainer = SimpleNamespace(epoch=0,save_dir=path,last=path/'weights/last.pt',best=path/'weights/best.pt')
                self.callback(self.trainer)
                (path/'args.yaml').write_text('epochs: 80', encoding='utf-8')
        fake_ultralytics = SimpleNamespace(YOLO=FakeYOLO,settings=SimpleNamespace(update=lambda args:None))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root/'data'; data.mkdir()
            for name in wf.META_FILES:
                (data/name).write_text('{}',encoding='utf-8')
            wf.write_json(data/'dataset_summary.json', {'source_sha256':{'dataset':'x','splits':'y'},'manifest_sha256':'m'})
            pretrained = root/'yolo11n.pt'; pretrained.write_bytes(b'fake pretrained')
            with patch.dict(sys.modules, {'ultralytics':fake_ultralytics}), \
                 patch.object(wf,'preflight'), patch.object(wf,'runtime_info',return_value={'gpu':'fake'}), \
                 patch.object(wf.subprocess,'run',return_value=SimpleNamespace(stdout='fake==1\n')):
                run = wf.train(data,root/'drive',root/'runs',pretrained,'tester',approved=True,labels_reviewed=True)
            self.assertEqual(wf.read_json(run/'status.json')['state'],'trained')
            self.assertEqual((run/'final/best.pt').read_bytes(), b'best weights')
            self.assertTrue((run/'latest_checkpoint.json').exists())
            self.assertEqual(wf.collect_results(root/'drive')[0]['member'],'tester')

    def test_resume_guard_and_resume_arguments_without_training(self):
        from workflow_fixture import prepared_dataset, evaluation_run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=prepared_dataset(root)
            parent=evaluation_run(root,data)
            wf.write_json(parent/'status.json',{'state':'interrupted'})
            source=root/'checkpoint_source'; source.mkdir()
            for name in ('last.pt','best.pt','results.csv'):
                (source/name).write_bytes(name.encode())
            wf.save_checkpoint(SimpleNamespace(epoch=0,last=source/'last.pt',best=source/'best.pt',save_dir=source),parent)
            checkpoint_state={'epoch':0,'optimizer':{},'train_args':{'data':str(data/'fire_v001_colab.yaml')}}
            calls=[]
            class ResumeModel:
                def __init__(self, path):
                    self.ckpt=checkpoint_state
                def add_callback(self, event, callback):
                    self.callback=callback
                def train(self, **args):
                    calls.append(args)
                    dest=Path(args['save_dir'])
                    assert (dest/'weights/best.pt').read_bytes()==b'best.pt'
                    assert args['resume'] is True
                    self.trainer=SimpleNamespace(save_dir=dest,epoch=1,last=dest/'weights/last.pt',best=dest/'weights/best.pt')
                    self.callback(self.trainer)
            fake=SimpleNamespace(YOLO=ResumeModel,settings=SimpleNamespace(update=lambda args:None))
            with patch.dict(sys.modules,{'ultralytics':fake}), patch.object(wf,'preflight'), \
                 patch.object(wf,'runtime_info',return_value={}), \
                 patch.object(wf.subprocess,'run',return_value=SimpleNamespace(stdout='')):
                child=wf.train(data,root/'drive',root/'runs',None,'resume',approved=True,labels_reviewed=True,resume_from=parent)
                self.assertEqual(wf.read_json(child/'run.json')['parent_run'],str(parent))
                self.assertTrue((child/'final/best.pt').exists())
                checkpoint_state['optimizer']=None
                with self.assertRaisesRegex(ValueError,'not resumable'):
                    wf.train(data,root/'drive',root/'runs',None,'bad',approved=True,labels_reviewed=True,resume_from=parent)
                checkpoint_state['optimizer']={}
                checkpoint_state['epoch']=79
                with self.assertRaisesRegex(ValueError,'already reached'):
                    wf.train(data,root/'drive',root/'runs',None,'complete',approved=True,labels_reviewed=True,resume_from=parent)
            self.assertEqual(len(calls),1)


if __name__ == '__main__':
    unittest.main()
