import unittest
import workflow_fixture
from fire_policy import filter_predictions
from tune_fire_policy import guard, metrics, BASELINE


def box(x=0,y=0,score=.8,cls=1):
    return dict(class_id=cls,score=score,xyxy=[x,y,x+10,y+10])


POLICY=dict(strong_conf=.25,weak_conf=.22,weak_support='near_strong_fire',
            support_radius=.4,duplicate_iou=.45,smoke_conf=.25)


class FirePolicyTests(unittest.TestCase):
    def test_low_fire_alone_cannot_generate_new_alarm(self):
        self.assertEqual(filter_predictions([box(score=.23)],POLICY,(100,100)),[])

    def test_smoke_alone_does_not_create_fire(self):
        self.assertEqual(filter_predictions([box(score=.9,cls=0)],POLICY,(100,100)),[0])

    def test_weak_candidate_near_strong_fire_is_kept(self):
        self.assertEqual(filter_predictions([box(),box(20,score=.23)],POLICY,(100,100)),[0,1])

    def test_far_weak_candidate_is_rejected(self):
        self.assertEqual(filter_predictions([box(),box(80,80,score=.23)],POLICY,(100,100)),[0])

    def test_duplicate_suppression_uses_score_and_keeps_classes_separate(self):
        predictions=[box(score=.5),box(score=.8),box(score=.9,cls=0)]
        self.assertEqual(filter_predictions(predictions,POLICY,(100,100)),[1,2])

    def test_background_and_weak_fire_counts(self):
        records=[dict(gt_fire=[],shape=[100,100],predictions=[box(score=.23)]),
                 dict(gt_fire=[[0,0,10,10]],shape=[100,100],predictions=[box()])]
        result=metrics(records,POLICY)
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,0,0))
        self.assertEqual(result['no_fire_image_alarm_fp'],0)

    def test_guard_rejects_losing_an_existing_positive_image(self):
        baseline=dict(tp=10,fp=5,image_alarm_tp=8,no_fire_image_alarm_fp=2)
        current=dict(tp=10,fp=3,image_alarm_tp=8,no_fire_image_alarm_fp=1,lost_baseline_positive_images=1)
        self.assertFalse(guard(current,baseline))

    def test_additional_smoke_support_requires_a_real_fire_candidate(self):
        policy=dict(POLICY,strong_conf=.3,weak_support='fire_or_smoke',support_smoke_conf=.5,
                    support_smoke_coverage=.5)
        preds=[box(score=.23),box(score=.9,cls=0)]
        self.assertEqual(filter_predictions(preds,policy,(100,100)),[0,1])
        self.assertEqual(filter_predictions([preds[1]],policy,(100,100)),[0])


if __name__=='__main__':
    unittest.main()
