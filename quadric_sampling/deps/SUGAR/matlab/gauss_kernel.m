function [K,sigma] = gauss_kernel(data1,data2, varargin)
% DEGREES    compute degree and sparsity estimates of data   https://arxiv.org/abs/1802.04927
%   Authors: Ofir Lindenbaum, Jay S. Stanely III.
% 
% Usage:
%         K = degrees(data1, data2, varargin) Build a kernel over data1,
%         data2
%         [K,sigma] = degrees(data1, data2, varargin) Return kernel
%         bandwidth sigma
%
% Input: 
%       data1           
%               N x D Data matrix. N rows are measurements, D columns are features.
%                   Accepts: 
%                       numeric
%       data2       If data2 == data1, then a traditional kernel is built.        
%               N x D Data matrix. N rows are measurements, D columns are features.
%                   Accepts: 
%                       numeric
%   varargin: 
%       sigma      (default = 'knn')
%               Gaussian kernel bandwidth. 
%                   Accepts:    
%                        'std'- standard deviation of the distances
%                        'knn' - adaptive bandwidth,eg  kth neighbor distance
%                        'minmax'-  min-max on the distance matrix       
%                        'median'- the median of the distances      
%                         function handle - @(d) f(d) = scalar or N-length 
%                                   vector where d is an NxN distance matrix.    
%                         scalar - pre-computed bandwidth
%
%       k               (default = 5)
%               k-Nearest neighbor distance to use if sigma = 'knn'
%                   Accepts:
%                       positive scalars
%       a               (default = 2)
%               Alpha-kernel decay parameter. 2 is Gaussian kernel.
%                   Accepts:
%                       positive scalars
%       fac             (default = 1)
%               Rescale kernel bandwidth
%                   Accepts:
%                       positive scalars
% Output:
%       K
%               Kernel over data1, data2
%       sigma
%               N x 1 (adaptive) or scalar (constant) estimated kernel bandwidth

[data1, data2, sigma, a, k, fac] = init(data1, data2, varargin{:});
D=pdist2(data1, data2);
N=size(D,2);

if strcmp(sigma,'minmax')
    MinDv=min(D+eye(size(D))*10^15);
    eps_val=(max(MinDv));
    sigma=2*(eps_val)^2;
elseif strcmp(sigma,'median')
    sigma=median(median(D));
elseif strcmp(sigma, 'std')
    sigma=std(mean(D));
elseif strcmp(sigma, 'knn')
    knnDST = sort(D);
    sigma = knnDST(k+1,:);
elseif isscalar(sigma)
    sigma = sigma;
elseif isa(sigma, 'function_handle')
    sigma = sigma(D);
end
sigma = sigma*fac;
K = bsxfun(@rdivide, D, sigma);
%Compute kernel elements
K=exp(-K.^a); 
K(isnan(K)) = 0;
K(K<1e-3) = 0;
if size(K,1)==size(K,2)
    K = (K+K')/2;
end
end

function [data1,data2, sigma, a, k, fac] = init(data, varargin)
    % helpers
    function tf = check_sigma(passed)
    % check that diffusion sigma is correct type
        valid_sigmas = {'std','knn', 'minmax', 'median'};
        if isscalar(passed) || isa(passed, 'function_handle') || ... %user supplied sigmas 
            any(cellfun(@(x) strcmp(passed,x), valid_sigmas))  % predefined sigma options
            tf = true;
        else
            tf = false;
        end
    end
    scalarPos = @(x) isscalar(x) && (x>0);

    % defaults
    default.sigma = 'knn';
    default.k = 5;
    default.a = 2;
    default.fac = 1;
    %parser configuration
    persistent p
    if isempty(p)
        p = inputParser;
        addRequired(p, 'data1', @isnumeric);
        addRequired(p, 'data2', @isnumeric);
        addParameter(p, 'sigma', default.sigma,@check_sigma);
        addParameter(p, 'k', default.k, scalarPos);
        addParameter(p, 'a', default.a, scalarPos);
        addParameter(p,'fac', default.fac,scalarPos);
    end
    %parse
    parse(p, data, varargin{:})
    data1 = p.Results.data1;
    data2 = p.Results.data2;
    sigma = p.Results.sigma;
    k = p.Results.k;
    a = p.Results.a;
    fac = p.Results.fac;
end