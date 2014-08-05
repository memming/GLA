function [L, dL, ddL] = neglogli_poissGLM(x, y, wts, fnlin)
% Negative log-likelihood for Poisson GLM
% [L, dL, ddL] = neglogli_poissGLM(x, y, wts, fnlin)
% 
% INPUT
%   x: (N x m) independent variable
%   y: (N x 1) dependent variable
%   wts: (m x 1) weights
%   fnlin: @(x)->(f,df,ddf) func handle for nonlinearity 
%	   (must return f, df and ddf; derivatives of f)
%
% OUTPUT
%   L: (1 x 1) negative log-likelihood
%  dL: (m x 1) gradient of L wrt weights
% ddL: (m x m) Hessian of L wrt weights
%
% See Also: neglogli_poissGLM_zaso

m = numel(wts);
xproj = x*wts;

if isequal(fnlin, @expfun) || isequal(fnlin, @exp) || strcmpi(exp, 'exp')
    f = exp(xproj);

    switch nargout
	case 1
	    L = -y' * xproj + sum(f);
	case 2
	    L = -y' * xproj + sum(f);
	    dL = x' * (f - y);
	case 3
	    L = -y' * xproj + sum(f);
	    dL = x' * (f - y);
	    ddL = x' * bsxfun(@times, x, f);
    end
    return
end

switch nargout
    case 1
        f = fnlin(xproj);
        L = -y'*log(f) + sum(f);
    case 2
        [f,df] = fnlin(xproj);
        L = -y'*log(f) + sum(f);
	dL = x'*((1 - y./f) .* df);
    case 3
        [f,df,ddf] = fnlin(xproj);
        L = -y'*log(f) + sum(f);
        yf = y./f;
	dL = x'*((1 - yf) .* df);
	ddL = bsxfun(@times,ddf.*(1-yf)+df.*(y./f.^2.*df) ,x)'*x;
end
