function tests = testGaussianLogLikelihood
    import gla.*
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    d = 10;
    N = 1000;
    testCase.TestData.smallRnd.x = randn(d, N);
    testCase.TestData.smallRnd.y = randn(N, 1);
    testCase.TestData.smallRnd.w = randn(d, 1);
    testCase.TestData.smallRnd.h = 0.01;
end

function test_ll_w_eq_h(testCase)
    x = testCase.TestData.smallRnd.x;
    y = testCase.TestData.smallRnd.y;
    w = testCase.TestData.smallRnd.w;
    h = testCase.TestData.smallRnd.h;
    L1 = gla.GaussianLogLikelihood.loglikelihood_w(x, y, w, h);
    L2 = gla.GaussianLogLikelihood.loglikelihood_h(x, y, w, h);
    testCase.verifyEqual(L1, L2);
end

function test_ll_w_grad(testCase)
end
