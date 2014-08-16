% Benchmark which optimization algorithm is faster
% Depending on the size of the input dimension, sometimes not computing
% the Hessian is faster, and zaso mini-batch might be better for some limit

addpath('zaso');
addpath('nlfuns');
addpath('minFunc2012/minFunc/');

%% Hold the total number of samples constant (constant memory to hold X)
% If I have 16G of memory, 8 bytes per double, 2.1e9
nList = 10;
mN = 1e8;
% nList = 5; mN = 1000; % warm setting
mList = round(logspace(0, 4, nList));
NList = ceil(mN ./ mList);

nMethods = 3; % MAGIC
methodNames = {'Newton', 'CG', 'Newton (no zaso)'};

for kList = 1:nList
    m = mList(kList);
    N = NList(kList);
    fprintf('== %d/%d (m=%d, N=%d)==\n', kList, nList, m, N);

    %% Generate data
    wTrue = sin((1:m)'/10)/10;
    x = randn(m, N);
    bTrue = -3;
    fnlin = @expfun;
    lambda = fnlin(wTrue' * x + bTrue);
    y = poissrnd(lambda);
    w0 = zeros(numel(wTrue)+1, 1);
    fprintf('Mean [%f], Sparsity [%f]\n', mean(y), nnz(y)/N);

    %% fminunc with Hessian
    tic;
    zaso = encapsulateRaw([x; ones(1,N)]', y', [], [], true); 

    optimOpts = optimoptions(@fminunc, ...
	'GradObj', 'on', 'Hessian', 'on', 'Display', 'off');
    [wOpt1, nLL1, exitflag1, output1] = fminunc(...
	@(w) neglogli_poissGLM_zaso(w, zaso, fnlin), w0, optimOpts);
    t1 = toc;
    fprintf('Newton [%f sec] # function call [%d], # of iterations [%d]\n', ...
	t1, output1.funcCount, output1.iterations);
    result(kList, 1).t = t1;
    result(kList, 1).output = output1;

    %% fminunc CG (medium-scale)
    tic;
    zaso = encapsulateRaw([x; ones(1,N)]', y', [], [], true); 

    optimOpts = optimset('LargeScale', 'off', ...
	'GradObj', 'on', 'Hessian', 'off', 'Display', 'off');
    [wOpt13, nLL13, exitflag13, output13] = fminunc(...
	@(w) neglogli_poissGLM_zaso(w, zaso, fnlin), w0, optimOpts);
    t13 = toc;
    fprintf('CG [%f sec] # function call [%d], # of iterations [%d]\n', ...
	t13, output13.funcCount, output13.iterations);
    result(kList, 2).t = t13;
    result(kList, 2).output = output13;

    %% fminunc with Hessian, no zaso
    tic;
    optimOpts = optimoptions(@fminunc, ...
	'GradObj', 'on', 'Hessian', 'on', 'Display', 'off');
    x1 = [x; ones(1,N)]';
    [wOpt3, nLL3, exitflag3, output3] = fminunc(...
	@(w) neglogli_poissGLM(x1, y', w, fnlin), w0, optimOpts);
    t3 = toc;
    fprintf('no zaso [%f sec] # function call [%d], # of iterations [%d]\n', ...
	t3, output3.funcCount, output3.iterations);
    result(kList, 3).t = t3;
    result(kList, 3).output = output3;
end
beep; system('say Done'); beep;

%% Plot some stuff
semilogy([result(:, 1).t])
semilogy([result(:, 2).t])
hold all; semilogy([result(:, 3).t])
