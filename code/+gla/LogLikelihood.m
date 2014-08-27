classdef (Abstract) LogLikelihood
    % Any log-likelihood with parameters and hyperparameters
    % must implement this interface class to interact with GLA.
    %
    % Evaluates $ 1/N * sum_i log p(y_i|x_i,w,h)$ and provides its derivatives
    % with respect to $w$ and also $hyp$.
    %
    % x: (d x N) independent variable
    % y: (1 x N) dependent variable
    % w: (? x 1) model parameter (weight)
    % h: (? x 1) model hyperparameter

    properties (Abstract, Constant)
	weightSpec;
	hypSpec;
    end

    methods (Abstract, Static)
	% derivatives with respect to w
	[L, dL_w, ddL_w] = loglikelihood_w(x, y, w, h);

	% derivatives with respect to h
	[L, dL_h, ddL_h] = loglikelihood_h(x, y, w, h);
    end
end
