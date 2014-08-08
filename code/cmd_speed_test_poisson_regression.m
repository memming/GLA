% We compare various implementation of Poisson regressions

addpath('zaso');
addpath('nlfuns');

%% Generate fake data
m = 2000; % # of variables
N = 5000;
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
disp('>>> zaso, fminunc, Newton');
tic;
zaso = encapsulateRaw([x; ones(1,N)]', y', [], [], true); 

optimOpts = optimoptions(@fminunc, ...
    'GradObj', 'on', 'Hessian', 'on', 'Display', 'off');
[wOpt1, nLL1, exitflag1, output1] = fminunc(...
    @(w) neglogli_poissGLM_zaso(w, zaso, fnlin), w0, optimOpts);
t1 = toc;

fprintf('[%f sec] # function call [%d], # of iterations [%d]\n', ...
    t1, output1.funcCount, output1.iterations);

% BFGS
disp('>>> zaso, fminunc, medium-scale?');
tic;
zaso = encapsulateRaw([x; ones(1,N)]', y', [], [], true); 

%optimOpts = optimoptions(@fminunc, 'LargeScale', 'off', ...
%    'GradObj', 'on', 'Hessian', 'off', 'Display', 'off');
optimOpts = optimset('LargeScale', 'off', ...
    'GradObj', 'on', 'Hessian', 'off', 'Display', 'off');
[wOpt13, nLL13, exitflag13, output13] = fminunc(...
    @(w) neglogli_poissGLM_zaso(w, zaso, fnlin), w0, optimOpts);
t13 = toc;
fprintf('[%f sec] # function call [%d], # of iterations [%d]\n', ...
    t13, output13.funcCount, output13.iterations);

disp('>>> zaso, fminunc, BFGS? Quasi-Newton? Hessian off');
tic;
zaso = encapsulateRaw([x; ones(1,N)]', y', [], [], true); 

optimOpts = optimoptions(@fminunc, 'Algorithm', 'quasi-newton', ...
    'GradObj', 'on', 'Hessian', 'off', 'Display', 'off');
[wOpt14, nLL14, exitflag14, output14] = fminunc(...
    @(w) neglogli_poissGLM_zaso(w, zaso, fnlin), w0, optimOpts);
t14 = toc;
fprintf('[%f sec] # function call [%d], # of iterations [%d]\n', ...
    t14, output14.funcCount, output14.iterations);

% % PCG?
% disp('>>> zaso, fminunc, PCG??');
% tic;
% zaso = encapsulateRaw([x; ones(1,N)]', y', [], [], true); 
% optimOpts = optimoptions(@fminunc, 'Algorithm', 'trust-region', ...
%     'GradObj', 'on', 'Hessian', 'off', 'Display', 'off');
% [wOpt12, nLL12, exitflag12, output12] = fminunc(...
%     @(w) neglogli_poissGLM_zaso(w, zaso, fnlin), w0, optimOpts);
% t12 = toc;
% 
% fprintf('[%f sec] # function call [%d], # of iterations [%d], # of CG iterations [%d]\n', ...
%     t12, output12.funcCount, output12.iterations, output12.cgiterations);

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

%% minFunc, CG
disp('>>> zaso, minFunc, CG');
if exist('minFunc') ~= 2
    error('Need minFunc for testing!');
end
tic;
zaso = encapsulateRaw([x; ones(1,N)]', y'); 

% opts21 = struct('maxFunEvals', 1000, 'Method', 'cg', ...
%     'display', 'iter', 'optTol', 1e-6, 'progTol', 1e-8);
opts21 = struct('maxFunEvals', 1000, 'Method', 'cg', ...
    'display', 'none', 'optTol', 1e-6, 'progTol', 1e-8);
[wOpt21, nLL21, exitflag21, output21] = minFunc(...
    @(w)(neglogli_poissGLM_zaso(w, zaso, fnlin)), w0, opts21);

t21 = toc;

fprintf('[%f sec] # function call [%d], # of iterations [%d]\n', ...
    t21, output21.funcCount, output21.iterations);

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
