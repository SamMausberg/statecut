import json
import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from statecut.capture import CaptureSDPA


def tensors(n=8):
    q = torch.ones((1,2,1,4), dtype=torch.bfloat16)
    k = torch.ones((1,2,n,4), dtype=torch.bfloat16)
    v = torch.full((1,2,n,4), 16, dtype=torch.bfloat16)
    return q,k,v


def test_capture_restores_and_preserves_bits(tmp_path):
    fn = torch.nn.functional.scaled_dot_product_attention
    q,k,v = tensors()
    expected = fn(q,k,v)
    with CaptureSDPA(tmp_path/"probe", allow_cpu=True) as cap:
        got = torch.nn.functional.scaled_dot_product_attention(q,k,v)
        assert torch.equal(got.view(torch.int16),expected.view(torch.int16))
        assert len(cap.files) == 1
        with np.load(tmp_path/"probe"/cap.files[0], allow_pickle=False) as z:
            assert z["out"].dtype == np.uint16
            assert z["out"].shape == (1,2,1,4)
            assert json.loads(str(z["metadata"]))["returned_output_unchanged"]
    assert torch.nn.functional.scaled_dot_product_attention is fn


def test_masks_and_limits_reject_not_truncate(tmp_path):
    q,k,v = tensors()
    with CaptureSDPA(tmp_path/"probe",allow_cpu=True,max_rows=4) as cap:
        torch.nn.functional.scaled_dot_product_attention(q,k,v)
        assert not cap.files
        assert cap.skipped["max-rows-not-a-truncation"] == 1
    with CaptureSDPA(tmp_path/"probe2",allow_cpu=True) as cap:
        torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        mask=torch.tensor([[True,True,False,False,False,False,False,False]])
        torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        assert not cap.files
        assert cap.skipped["dropout-or-causal-semantics"] == 1
        assert cap.skipped["not-full-visible-zero-bias"] == 1


def test_exception_restores_and_nonempty_refused(tmp_path):
    fn = torch.nn.functional.scaled_dot_product_attention
    with pytest.raises(RuntimeError):
        with CaptureSDPA(tmp_path/"probe",allow_cpu=True):
            raise RuntimeError("abort")
    assert torch.nn.functional.scaled_dot_product_attention is fn
    with pytest.raises(FileExistsError):
        with CaptureSDPA(tmp_path/"probe",allow_cpu=True):
            pass


def test_grouped_query_mapping_is_recorded(tmp_path):
    q,k,v=tensors()
    k,v=k[:,:1],v[:,:1]
    with CaptureSDPA(tmp_path/"gqa",allow_cpu=True) as cap:
        actual=torch.nn.functional.scaled_dot_product_attention(q,k,v,enable_gqa=True)
        assert torch.equal(actual,torch.full_like(actual,16))
        with np.load(tmp_path/"gqa"/cap.files[0],allow_pickle=False) as z:
            meta=json.loads(str(z["metadata"]))
            assert meta["q_to_kv_group_size"] == 2 and meta["enable_gqa"]
            assert z["k"].shape[1] == 1 and z["q"].shape[1] == 2
