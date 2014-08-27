classdef GaussianLogLikelihood < gla.LogLikelihood
    properties (Constant)
	weightSpec = 'linear';
	hypSpec = 'sigma^2';
    end

    methods (Static) % implementing LogLikelihood interface
	% derivatives with respect to w
	function [L, dL_w, ddL_w] = loglikelihood_w(x, y, w, h)
	    N = size(x, 2);
	    residual = x' * w - y;
	    L = -residual' * residual / N / h / 2 - log(2*pi*h) / 2;
	    if nargout > 1
		dL_w = - x * residual / N / h;
	    end
	    if nargout > 2
		% TODO precompute x * x'? But what if x changes?
		% Also, what will you do when computing mini-batch?
		XX = x * x';
		ddL_w = - XX / N / h;
	    end
	end

	% derivatives with respect to h
	function [L, dL_h, ddL_h] = loglikelihood_h(x, y, w, h);
	    N = size(x, 2);
	    residual = x' * w - y;
	    mse = residual' * residual / N;
	    L = - mse/h/2 - log(2*pi*h) / 2;
	    if nargout > 1
		dL_h = mse / h^2 / 2 - 1 / 2 / h;
	    end
	    if nargout > 2
		ddL_h = - mse / h^3 + 1 / h^2;
	    end
	end
    end

    methods (Static) % fast computation tricks just for Gaussian case
    end
end
