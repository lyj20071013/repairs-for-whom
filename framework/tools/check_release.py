"""CPU validation and executable delivery report. Does not run GPU workloads."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from repair_eval.dataset import Dataset
from repair_eval.evaluate import evaluate
from repair_eval.io import write_json,sha256
from examples.selectors import sampled_gain,uniform


def main():
    checks=[]
    for folder in sorted((ROOT/"data").iterdir()):
        if (folder/"manifest.json").exists():
            checks.append(Dataset(folder).validate())
    ds=Dataset(ROOT/"data"/"alfworld-q7")
    matched=evaluate(ds,selector=sampled_gain,uses_rollouts=True,splits=3,replicates=100)
    if matched["selection_loss"]["mean"] != 0.0:
        raise RuntimeError("The example matched selector must have zero loss")
    ws=Dataset(ROOT/"data"/"webshop-q7")
    rand=evaluate(ws,selector=uniform,splits=3,replicates=100)
    if abs(rand["selection_loss"]["mean"]-rand["positive_control"]["mean"])>1e-12:
        raise RuntimeError("Uniform selection loss must equal the paired random control")
    test=subprocess.run([sys.executable,"-X","utf8","-m","unittest","discover","-s","tests","-v"],
                        cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
    if test.returncode:
        print(test.stdout+test.stderr)
        return test.returncode
    report={"python":sys.version.split()[0],"numpy":__import__("numpy").__version__,
            "pools":checks,"tests_passed":True,"test_output":test.stdout+test.stderr,
            "example_matched_loss":matched["selection_loss"]["mean"],
            "example_uniform_identity_error":abs(rand["selection_loss"]["mean"]-rand["positive_control"]["mean"]),
            "real_gpu_tested":False,"source_project_modified":False}
    rep=ROOT/"outputs/reproduction/reproduction.json"
    if rep.exists():
        original=json.loads(rep.read_text(encoding="utf-8"))
        report["paper_reproduction"]={"passed":original["passed"],"report_sha256":sha256(rep),
                                       "comparisons":original["comparisons"],"version_notices":original["version_notices"]}
    write_json(ROOT/"provenance"/"validation.json",report)
    print(json.dumps({"pools":len(checks),"tests_passed":True,"matched_loss":report["example_matched_loss"],
                      "paper_reproduction":report.get("paper_reproduction",{}).get("passed")},indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
