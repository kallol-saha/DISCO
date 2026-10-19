import numpy as np
import torch

from disco.policy import models as M


def test_roundtrip_and_loss(tmp_path):
    for kind, ch in [("grasp", 3), ("place", 4)]:
        m = M.build_policy(kind, ch)
        gp = torch.randn(8, 2, 8)
        M.fit_policy_normalizer(m, gp, kind)
        p = tmp_path / f"{kind}.pth"
        M.save_weights(m, str(p), 5)
        m2 = M.load_weights(M.build_policy(kind, ch), str(p))
        assert torch.equal(m2.workspace_normalizer, m.workspace_normalizer)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m2.cuda()(gp.cuda(), torch.randn(8, 4096, ch).cuda())
        assert torch.isfinite(loss)
    v = M.build_value().cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = v(pcd=torch.randn(4, 4096, 3).cuda(), actions=torch.randn(4, 2, 8).cuda(),
                 targets=torch.rand(4).cuda(), n_leaves=torch.ones(4).cuda())
    assert torch.isfinite(loss)


def test_inference_shapes():
    g = M.build_policy("grasp", 3).cuda().eval()
    p = M.build_policy("place", 3).cuda().eval()
    v = M.build_value().cuda().eval()
    pcd = np.random.randn(4096, 3).astype(np.float32)
    gr = M.sample_grasps(g, pcd, 5)
    pl = M.sample_places(p, pcd, gr)
    assert gr.shape == (5, 7) and pl.shape == (5, 7)
    acts = torch.stack([torch.cat([gr, torch.ones(5, 1)], 1), torch.cat([pl, torch.zeros(5, 1)], 1)], 1)
    q = M.score(v, pcd, acts)
    assert q.shape == (5,) and (q >= 0).all() and (q <= 1).all()
