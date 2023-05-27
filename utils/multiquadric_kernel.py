import torch
import gpytorch


class MultiquadricKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    
    def __init__(
        self,
        batch_shape=torch.Size(),
        lengthscale_prior=None,
        lengthscale_constraint=None,
        **kwargs
    ):
        super().__init__(
            has_lengthscale=self.has_lengthscale,
            batch_shape=batch_shape,
            **kwargs
        )
        
        if lengthscale_constraint is None:
            lengthscale_constraint = gpytorch.constraints.Positive()
            
        self.raw_lengthscale = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        self.lengthscale_constraint = lengthscale_constraint
        self.register_parameter(
            name="raw_lengthscale",
            parameter=self.raw_lengthscale
        )
        
        if lengthscale_prior is not None:
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda: self.lengthscale,
                lambda v: self._set_lengthscale(v),
            )
            
        self.lengthscale = self.lengthscale_constraint.transform(self.raw_lengthscale)
        
    def forward(self, x1, x2, diag=False, **params):
        lengthscale = self.lengthscale.unsqueeze(-1)
        x1_ = x1.div(lengthscale)
        x2_ = x2.div(lengthscale)
        if diag:
            return ((x1_ - x2_) ** 2).sum(dim=-1).sqrt()
        else:
            return ((x1_.unsqueeze(-2) - x2_.unsqueeze(-3)) ** 2).sum(dim=-1).sqrt()
