import torch

def test_symmetric_analytic_jacobian_and_hessian():
 torch.manual_seed(3);d,r,o=5,3,4;s=0.7;x=torch.randn(d,dtype=torch.double,requires_grad=True);U=torch.randn(d,r,dtype=torch.double);P=torch.randn(r,o,dtype=torch.double)
 def f(t): return s*((t@U).square()@P)
 auto=torch.autograd.functional.jacobian(f,x).T
 analytic=2*s*U@torch.diag(x.detach()@U)@P
 torch.testing.assert_close(analytic,auto,rtol=1e-9,atol=1e-10)
 c=torch.randn(o,dtype=torch.double)
 ah=torch.autograd.functional.hessian(lambda t:(f(t)*c).sum(),x)
 exact=2*s*(U*(P@c).unsqueeze(0))@U.T
 torch.testing.assert_close(exact,ah,rtol=1e-9,atol=1e-10)

def test_lora_adapter_hessian_is_zero():
 torch.manual_seed(4);x=torch.randn(5,dtype=torch.double,requires_grad=True);A=torch.randn(5,2,dtype=torch.double);B=torch.randn(2,4,dtype=torch.double);c=torch.randn(4,dtype=torch.double)
 h=torch.autograd.functional.hessian(lambda t:(((t@A)@B)*c).sum(),x)
 torch.testing.assert_close(h,torch.zeros_like(h))
