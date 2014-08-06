% We compare various implementation of Poisson regressions

addpath('zaso');
addpath('nlfuns');

%% Generate fake data
m = 100; % # of variables
N = 100000;
wTrue = sin((1:m)'/10)/10;
x = randn(m, N);
bTrue = -3;
fnlin = @expfun;
lambda = fnlin(wTrue' * x + bTrue);
y = poissrnd(lambda);
w0 = zeros(numel(wTrue)+1, 1);
fprintf('Mean [%f], Sparsity [%f]\n', mean(y), nnz(y)/N);

%%%%%%%%%%%%%%%%%%%%%%% Regression Time %%%%%%%%%%%%%%%%%%%%%%%

%% zaso version
% TODO why the hell do I need to transpose????
disp('>>> zaso, fminunc');
tic;
zaso = encapsulateRaw([x; ones(1,N)]', y'); 

optimOpts = optimoptions(@fminunc, ...
    'GradObj', 'on', 'Hessian', 'on', 'Display', 'off');
[wOpt1, nLL1, exitflag1, output1] = fminunc(...
    @(w) neglogli_poissGLM_zaso(w, zaso, fnlin), w0, optimOpts);
t1 = toc;

fprintf('[%f sec] # function call [%d], # of iterations [%d]\n', ...
    t1, output1.funcCount, output1.iterations);

%% minFunc, Newton
disp('>>> zaso, minFunc, Newton');
if exist('minFunc') ~= 2
    error('Need minFunc for testing!');
end
% addpath('~/pillowlab/lib/minFunc_2012/minFunc/');
tic;
zaso = encapsulateRaw([x; ones(1,N)]', y'); 

opts1 = struct('maxFunEvals', 100, 'Method', 'newton', 'display', 'none');
[wOpt2, nLL2, exitflag2, output2] = minFunc(...
    @(w)(neglogli_poissGLM_zaso(w, zaso, fnlin)), w0, opts1);

t2 = toc;

fprintf('[%f sec] # function call [%d], # of iterations [%d]\n', ...
    t2, output2.funcCount, output2.iterations);

%% Non-zaso version
disp('>>> no-zaso, fminunc');
tic;
optimOpts = optimoptions(@fminunc, ...
    'GradObj', 'on', 'Hessian', 'on', 'Display', 'off');
x1 = [x; ones(1,N)]';
[wOpt3, nLL3, exitflag3, output3] = fminunc(...
    @(w) neglogli_poissGLM(x1, y', w, fnlin), w0, optimOpts);
t3 = toc;

fprintf('[%f sec] # function call [%d], # of iterations [%d]\n', ...
    t3, output3.funcCount, output3.iterations);
