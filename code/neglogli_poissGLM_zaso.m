function [L,dL,ddL] = neglogli_poissGLM_zaso(wts, zaso, fnlin)
% Negative log-likelihood for Poisson GLM using ZASO
% [L,dL,ddL] = neglogli_poissGLM(wts, zaso, Y)
%
% Compute negative log-likelihood of data under Poisson regression model,
% plus gradient and Hessian
%
% INPUT
%   wts: (m x 1) regression weights
%   zaso: Zhe Abstract Sample Object
%   fnlin: @(x)->(f,df,ddf) func handle for nonlinearity 
%	   (must return f, df and ddf; derivatives of f)
%
% OUTPUT
%   L: (1 x 1) negative log-likelihood
%  dL: (m x 1) gradient of L wrt weights
% ddL: (m x m) Hessian of L wrt weights
%
% Example usage:
%   fminunc(@(w) neglogli_poissGLM_zaso(w, zaso, fnlin), wInit)
%
% See Also: neglogli_poissGLM

m = numel(wts);

if isequal(fnlin, @expfun) || isequal(fnlin, @exp) || strcmpi(exp, 'exp')
    % fast special computation for exponential nonlinearity
    % Note that this trick of comparing function handles doesn't work for ones
    % that are loaded from MAT files.
    switch nargout
	case 1
	    L = zasoFxysum(zaso, ...
		@(x,y) neglogli_poissGLM_sub_exp(x, y, wts, 1));
	case 2
	    r = zasoFxysum(zaso, ...
		@(x,y) neglogli_poissGLM_sub_exp(x, y, wts, 2));
	    L = r(1);
	    dL = r(2);
	case 3
	    r = zasoFxysum(zaso, ...
		@(x,y) neglogli_poissGLM_sub_exp(x, y, wts, 3));
	    L = r(1);
	    dL = r(1 + (1:m));
	    ddL = reshape(r(m+2:end), m, m);
	otherwise
	    error('Only up to 3 output arguments');
    end
    return
end

switch nargout
    case 1
        L = zasoFxysum(zaso, @(x,y) neglogli_poissGLM_sub(x, y, wts, fnlin, 1));
    case 2
        r = zasoFxysum(zaso, @(x,y) neglogli_poissGLM_sub(x, y, wts, fnlin, 2));
        L = r(1);
        dL = r(2);
    case 3
        r = zasoFxysum(zaso, @(x,y) neglogli_poissGLM_sub(x, y, wts, fnlin, 3));
        L = r(1);
        dL = r(1 + (1:m));
        ddL = reshape(r(m+2:end), m, m);
    otherwise
	error('Only up to 3 output arguments');
end

end % neglogli_poissGLM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = neglogli_poissGLM_sub(x, y, wts, fnlin, choice)
% choice = 1: only likelihood
%	   2: likelihood and gradient
%	   3: likelihood, graidient, and Hessian

m = numel(wts);
xproj = wts' * x;

switch choice
    case 1
        f = fnlin(xproj);
        L = -y'*log(f) + sum(f);
    case 2
        [f,df] = fnlin(xproj);
        L = zeros(m+1, 1);
        L(1) = -y'*log(f) + sum(f);
        L(1 + (1:m)) = x'*((1 - y./f) .* df);
    case 3
        [f,df,ddf] = fnlin(xproj);
        L = zeros(m^2+m+1, 1);
        L(1) = -y'*log(f) + sum(f);
        yf = y./f;
        L(1 + (1:m)) = x'*((1 - yf) .* df);
	H = bsxfun(@times,ddf.*(1-yf)+df.*(y./f.^2.*df) ,x)'*x;
        L(m+2:end) = H(:);
end
end % neglogli_poissGLM_sub

function L = neglogli_poissGLM_sub_exp(x, y, wts, choice)
% choice = 1: only likelihood
%	   2: likelihood and gradient
%	   3: likelihood, graidient, and Hessian
%
% special case for exponential nonlinearity (the CANONICAL LINK)

m = numel(wts);
xproj = wts' * x; % row vector
f = exp(xproj);

switch choice
    case 1
        L = -xproj * y' + sum(f);
    case 2
        L = zeros(m+1, 1);
        L(1) = -xproj * y' + sum(f);
        L(1 + (1:m)) = (f - y) * x';
    case 3
        L = zeros(m^2+m+1, 1);
        L(1) = -xproj * y' + sum(f);
        L(1 + (1:m)) = (f - y) * x';
	H = bsxfun(@times, f, x) * x';
	% H = zeros(m); % devectorizing is super slow... :'(
	% for k = 1:size(x, 2)
	%     H = H + x(:, k) * f(k) * x(:, k)';
	% end
        L(m+2:end) = H(:);
end
end % neglogli_poissGLM_sub_exp
