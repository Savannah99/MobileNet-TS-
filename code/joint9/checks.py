"""Verify budgets, within-bin selection and exact compatibility with Pedi8 code."""
from base import *
from vision_joint import select_joint
import importlib.util

setup()
rng=np.random.default_rng(77)
prob=rng.random(165).astype("float32");h=rng.random((165,64)).astype("float32")
for k in (4,8):
    for beta in (0.,BETA):
        ix=select_joint(prob,h,k,beta)
        assert ix.shape==(10,k)
        assert np.array_equal((ix-np.arange(10)[:,None]*16)//(16//k),
                              np.broadcast_to(np.arange(k),(10,k)))
        assert (np.diff(ix,axis=1)>0).all()
    assert np.array_equal(select_joint(prob,h,k,0),indices("tsplus",len(prob),k,prob))
# Compare against previously run Pedi8 indices, both val and test already authorized/observed.
# No new test images or labels enter the new selector definition.
p=ROOT/"outputs/pedi_followup_queue/02_joint_selection/rich/video13.npz"
with np.load(p) as z:
    actual=select_joint(z["confidence"],z["features"],8,BETA)
with np.load(ROOT/"outputs/pedi_followup_queue/02_joint_selection/eval/selected/ts8/seed0/video01.npz") as z:
    assert z["selected_indices"].shape[1]==8
# Compare a validation example to the saved previous algorithm without importing its support module.
module=ROOT/"code/conference/followup_queue/joint_vision.py"
import ast
tree=ast.parse(module.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="select_joint")
namespace={"np":np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(module),"exec"),namespace)
with np.load(p) as z:
    assert np.array_equal(actual,namespace["select_joint"](z["confidence"],z["features"],BETA))
for k in (4,8):
    x=np.zeros((16,64),np.float32);x[:,0]=1
    x[16//k+1,0]=0;x[16//k+1,1]=1
    p0=np.full(16,.5)
    assert select_joint(p0,x,k,BETA)[0,1]==16//k+1
save_json(RUN/"module_checks.json",{"state":"complete","budgets":[4,8],
    "beta0_identity":True,"positive_novelty_effect":True,"prior_pedi8_validation_indices_exact":True,
    "selector_sha256":sha256(Path(__file__).parent/"vision_joint.py")})
print("MODULE_CHECKS_COMPLETE")
