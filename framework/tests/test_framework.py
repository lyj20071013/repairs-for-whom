import tempfile
import unittest
from pathlib import Path
import numpy as np
from repair_eval.dataset import Dataset
from repair_eval.evaluate import evaluate, weights
from repair_eval.io import read_json, write_json, read_jsonl, write_jsonl, loads, sha256, fingerprint
from repair_eval.stats import holm, bootstrap
from repair_eval.replay import plan_replay, run_replay, attach_replay, load_plan
from tests.helpers import tiny_bundle
from examples.selectors import sampled_gain, uniform


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.ds=tiny_bundle(self.root/"bundle")
    def tearDown(self):
        self.tmp.cleanup()

    def test_known_crossfit_prevents_winners_curse(self):
        # a has [1,0], b and original [0,0]. In-sample max is .5, but
        # held-out matched value is .25 after exact tie averaging.
        r=evaluate(self.ds,selector=sampled_gain,uses_rollouts=True,budget=1,splits=4,replicates=20)
        self.assertAlmostEqual(r["values"]["mean"],.25)
        self.assertEqual(r["selection_loss"]["mean"],0.)

    def test_plugin_gets_no_evaluation_outcomes(self):
        calls=[]
        def choose(task,context):
            self.assertNotIn("outcomes",task)
            self.assertEqual(set(context),{"seed","split","fold","objective","metric","budget","selection"})
            self.assertEqual(set(context["selection"]["a"]),{"alternative","original"})
            calls.append(1)
            return sampled_gain(task,context)
        evaluate(self.ds,selector=choose,uses_rollouts=True,budget=1,splits=3,replicates=20)
        self.assertEqual(len(calls),6)

    def test_label_free_called_once(self):
        calls=[]
        def choose(task,context):
            calls.append(context)
            self.assertEqual(set(context),{"seed"})
            return "a"
        r=evaluate(self.ds,selector=choose,budget=1,splits=2,replicates=20)
        self.assertEqual(len(calls),1)
        self.assertAlmostEqual(r["values"]["mean"],.5)
        self.assertAlmostEqual(r["selection_loss"]["mean"],-.25)

    def test_predictions_missing_is_error(self):
        with self.assertRaises(ValueError):
            evaluate(self.ds,predictions={"bundle_id":self.ds.manifest["bundle_id"],"choices":{}},budget=1)

    def test_predictions_wrong_version_is_error(self):
        with self.assertRaises(ValueError):
            evaluate(self.ds,predictions={"bundle_id":"old","choices":{"t0":"a"}},budget=1)

    def test_explicit_abstain(self):
        p={"bundle_id":self.ds.manifest["bundle_id"],"choices":{"t0":None}}
        r=evaluate(self.ds,predictions=p,budget=1,splits=2,replicates=20)
        self.assertEqual(r["values"]["mean"],0.)
        with self.assertRaises(ValueError):
            evaluate(self.ds,predictions=p,budget=1,objective="post")

    def test_invalid_choice(self):
        for c in ["missing",{}, {"a":2.},{"a":float("nan")},{"a":-1.,"b":2.}]:
            with self.subTest(c=c),self.assertRaises(ValueError):
                weights(c,["a","b"])

    def test_hash_tamper(self):
        (self.ds.path/"tasks.jsonl").write_text("{}\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            Dataset(self.ds.path)

    def test_manifest_tamper(self):
        m=read_json(self.ds.path/"manifest.json");m["deployer"]="wrong"
        write_json(self.ds.path/"manifest.json",m)
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            Dataset(self.ds.path)

    def test_incomplete_rollouts_fail_even_with_new_hash(self):
        f=self.ds.path/"outcomes/d.jsonl"
        write_jsonl(f,read_jsonl(f)[:-1])
        m=read_json(self.ds.path/"manifest.json");m["files"]["outcomes/d.jsonl"]=sha256(f)
        m.pop("bundle_id");m["bundle_id"]=fingerprint(m);write_json(self.ds.path/"manifest.json",m)
        with self.assertRaisesRegex(ValueError,"coverage"):
            Dataset(self.ds.path)

    def test_strict_json(self):
        for s in ['{"a":1,"a":2}','{"a":NaN}']:
            with self.assertRaises(ValueError):loads(s)

    def test_holm_pending_family(self):
        t=holm({"a":{"p":.02},"b":{"p":.03}},5)
        self.assertFalse(t["a"]["holm_reject"])
        self.assertAlmostEqual(t["a"]["p_holm"],.10)
        with self.assertRaises(ValueError):holm(t,1)

    def test_bootstrap_pairs_and_finite(self):
        b=bootstrap([.1]*8,replicates=30)
        self.assertTrue(np.allclose(b["ci"],[.1,.1]))
        with self.assertRaises(ValueError):bootstrap([np.nan])

    def test_replay_resume_and_attach(self):
        plan=self.root/"plan.json";out=self.root/"run"
        plan_replay(self.ds,"new","test-model",2,"B",plan)
        with self.assertRaisesRegex(ValueError,"pending"):
            run_replay(plan,"tests.helpers:fake_backend",{"partial":True},out)
        self.assertEqual(len(read_jsonl(out/"results.jsonl")),3)
        with self.assertRaisesRegex(ValueError,"Cannot resume"):
            run_replay(plan,"tests.helpers:fake_backend",{"partial":False},out)
        with self.assertRaisesRegex(ValueError,"pending"):
            run_replay(plan,"tests.helpers:fake_backend",{"partial":True},out)
        run_replay(plan,"tests.helpers:fake_backend",{"partial":True},out)
        again=run_replay(plan,"tests.helpers:fake_backend",{"partial":True},out)
        self.assertTrue(again["resumed_without_backend"])
        attach_replay(self.ds,plan,out,self.root/"expanded")
        new=Dataset(self.root/"expanded")
        self.assertEqual(new.manifest["executors"]["new"]["samples"],2)
        self.assertNotIn("new",self.ds.rollouts)

    def test_incomplete_replay_not_attached(self):
        plan=self.root/"plan.json";out=self.root/"run"
        plan_replay(self.ds,"new","test-model",2,"B",plan)
        with self.assertRaises(ValueError):run_replay(plan,"tests.helpers:fake_backend",{"partial":True},out)
        with self.assertRaisesRegex(ValueError,"incomplete"):
            attach_replay(self.ds,plan,out,self.root/"expanded")

    def test_replay_plan_tamper(self):
        plan=self.root/"plan.json"
        plan_replay(self.ds,"new","test-model",2,"B",plan)
        p=read_json(plan);p["jobs"][0]["seed"]=99;write_json(plan,p)
        with self.assertRaises(ValueError):load_plan(plan)


if __name__=="__main__":unittest.main()
