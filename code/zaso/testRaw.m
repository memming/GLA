%%
N = 20000;
X = randn(4, N);
wTrue = [3 2 1 -1]';
Y = wTrue' * X + 0.1 * randn(1, N);

zaso = encapsulateRaw(X, Y);

NTest = 1000;
XTest = randn(4, NTest);
YTest = wTrue' * XTest + 0.1 * randn(1, NTest);

zasoTest = encapsulateRaw(XTest, YTest);

%% Obtaining moments of data
EX = zasoFxsum(zaso, @(x) sum(x,2)) / zaso.N;
EY = zasoFysum(zaso, @(y) sum(y,2)) / zaso.N;
EXX = zasoFxsum(zaso, @(x) x * x') / zaso.N;
EXY = zasoFxysum(zaso, @(x,y) x * y') / zaso.N; % for 1-D y

%% Basic linear regression example
XX = zasoFxsum(zaso, @(x) x * x');
XY = zasoFxysum(zaso, @(x,y) x * y');

w = XX \ XY;

mseTest = zasoFxysum(zasoTest, @(x,y) sum((w' * x - y).^2)) / zasoTest.N
prediction = zasoFx(zasoTest, @(x) w' * x);

%% Compute quantities in batch at once (faster than calling them separately)
[rsum, ragg] = zasoFarray(zaso, ...
		{@(x,y) sum(x,2), @(x,y) sum(y,2)}, ...
		{@(x,y) w' * x, @(x,y) (w' * x).^2});

%% Quadratic basis (1, x, x^2) regression example
% no cross-terms here
fX = @(X) [ones(1, size(X, 2)); X; X.^2];
Y = 3 + wTrue' * X + 0.2 * wTrue' * X.^2 + 0.1 * randn(1, N);
zaso = encapsulateRaw(X, Y, fX);
YTest = 3 + wTrue' * XTest + 0.2 * wTrue' * XTest.^2 + 0.1 * randn(1, NTest);
zasoTest = encapsulateRaw(XTest, YTest, fX);

[rsum, ragg] = zasoFarray(zaso, {@(x,y) x * x', @(x,y) x * y'}, {});
XX = rsum{1};
XY = rsum{2};
w = XX \ XY;

mseTest = zasoFxysum(zasoTest, @(x,y) sum((w' * x - y).^2), 2) / zasoTest.N
prediction = zasoFx(zasoTest, @(x) w' * x);

%% subindexing test
zasoFx(zaso, @(x) sum(x), 1:10)
zasoFx(zaso, @(x) sum(x), 1)
zasoFxsum(zaso, @(x) sum(x), 1)

%% Bulk-mode
isBulkProcess = true;
zasoB = encapsulateRaw(X, Y, fX, [], isBulkProcess);
assert(any(any(zasoFxsum(zaso, @(x) x * x') == zasoFxsum(zasoB, @(x) x * x'))));
