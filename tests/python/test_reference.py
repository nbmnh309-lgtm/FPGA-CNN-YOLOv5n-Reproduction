import numpy as np
import pytest
import torch
from reproduction.reference import *


def test_quant_boundaries_and_rounding():
    d = Domain(0.5, 100)
    np.testing.assert_array_equal(d.quantize([-100, -.75, -.25, .25, .75, 100]), [0, 98, 100, 100, 102, 255])
    for x in (np.zeros(10), np.ones(10)*-4, np.arange(-10, 20)):
        q = calibrate(x); assert np.max(np.abs(q.dequantize(q.quantize(x))-x)) <= q.scale/2 + 1e-9
    for bad in (0, -1, float('nan')):
        with pytest.raises(ValueError): Domain(bad, 0)
    with pytest.raises(ValueError): calibrate([float('nan')])


def test_mqda_random_and_saturation():
    rng = np.random.default_rng(42); a=rng.integers(0,256,(10000,4)); b=rng.integers(0,256,a.shape)
    da,db,out=Domain(.013,91),Domain(.087,137),Domain(.09,120)
    target=da.dequantize(a)+db.dequantize(b)
    q=mqda_add(a,b,da,db,out)
    np.testing.assert_array_equal(q,out.quantize(target))
    aligned=np.mean((out.dequantize(q)-target)**2)
    naive=np.mean((out.dequantize(naive_add(a,b,out))-target)**2)
    assert aligned < naive
    cat=mqda_cat(a,b,da,db,out)
    np.testing.assert_array_equal(cat[:,:4],out.quantize(da.dequantize(a)))
    np.testing.assert_array_equal(cat[:,4:],out.quantize(db.dequantize(b)))
    assert mqda_add([0],[0],da,db,out,True)[0] == out.zero


def test_q16_negative_ties():
    np.testing.assert_array_equal(requant_q16([-3,-1,1,3],32768,100),[98,99,101,102])
    np.testing.assert_array_equal(requant_q16([-10000,10000],65536,0),[0,255])


def test_dmp_all_weights_and_all_activation_codes():
    a=np.arange(-128,128)[:,None]; b=np.arange(-128,128)[None,:]
    for c in range(-128,256):
        hi,lo=dmp(a,b,c)
        np.testing.assert_array_equal(hi,np.broadcast_to(a*c,(256,256)))
        np.testing.assert_array_equal(lo,np.broadcast_to(b*c,(256,256)))


@pytest.mark.parametrize('kernel,stride,padding',[(1,1,0),(3,1,1),(3,2,1)])
def test_conv_against_torch(kernel,stride,padding):
    rng=np.random.default_rng(8); x=rng.integers(0,256,(7,9,3)); w=rng.integers(-128,128,(5,3,kernel,kernel)); bias=rng.integers(-1000,1000,5)
    actual=conv(x,w,bias,stride,padding,128)
    expected=torch.nn.functional.conv2d(torch.tensor((x-128).transpose(2,0,1)[None],dtype=torch.float64),torch.tensor(w,dtype=torch.float64),torch.tensor(bias,dtype=torch.float64),stride,padding)
    np.testing.assert_array_equal(actual,expected[0].permute(1,2,0).numpy())


def test_spatial_ops_against_torch_and_focus_order():
    x=np.arange(6*8*3).reshape(6,8,3)
    f=focus(x); assert tuple(f[0,0]) == tuple(np.concatenate((x[0,0],x[1,0],x[0,1],x[1,1])))
    np.testing.assert_array_equal(upsample(x)[::2,::2],x)
    t=torch.tensor(x.transpose(2,0,1)[None],dtype=torch.float64); stages=[t]
    for _ in range(3): stages.append(torch.nn.functional.max_pool2d(stages[-1],5,1,2))
    np.testing.assert_array_equal(sppf(x),torch.cat(stages,1)[0].permute(1,2,0).numpy())
    with pytest.raises(ValueError): focus(x[:5])


def test_bank_ownership_and_overlap():
    for n in range(1,30):
        for latencies in [(2,9,3),(9,2,7),(4,4,4)]:
            pp=buffer_schedule(n,*latencies); serial=buffer_schedule(n,*latencies,False)
            assert pp['cycles']<=serial['cycles']
            assert serial['cycles']==n*sum(latencies)
            for i,e in enumerate(pp['events']):
                assert e['read'][1]<=e['compute'][0]<=e['compute'][1]<=e['write'][0]
                if i>=2: assert pp['events'][i-2]['write'][1]<=e['read'][0]
    assert buffer_schedule(10,5,20,5)['cycles'] < buffer_schedule(10,5,20,5,False)['cycles']


def test_instruction_memory_integration():
    x=np.arange(8*8*3).reshape(8,8,3)%256; d=Domain(1,0)
    n=MemoryNPU(); n.memory[0x1000]=x
    w=np.ones((2,12,1,1),dtype=np.int64)
    instructions=[{'op':'focus','src':[0x1000],'dst':0x2000}, {'op':'conv11','src':[0x2000],'dst':0x3000,'args':{'weights':w}}, {'op':'upsample','src':[0x3000],'dst':0x4000}, {'op':'add','src':[0x4000,0x4000],'dst':0x5000,'args':{'da':d,'db':d,'out':d}}, {'op':'sppf','src':[0x5000],'dst':0x6000}, {'op':'cat','src':[0x6000,0x6000],'dst':0x7000,'args':{'da':d,'db':d,'out':d}}]
    expected=np.clip(upsample(conv(focus(x),w))*2,0,255); expected=sppf(expected); expected=np.concatenate((expected,expected),axis=-1)
    np.testing.assert_array_equal(n.run(instructions),expected)
    assert len(n.transfers)==14
