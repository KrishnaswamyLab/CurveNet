function [d_hat,s_hat,sigma] = degrees(data, varargin)
% DEGREES    compute degree and sparsity estimates of data   https://arxiv.org/abs/1802.04927
%   Authors: Ofir Lindenbaum, Jay S. Stanely III.
% 
% Usage:
%         d_hat = degrees(data, varargin) Estimate manifold degrees d_hat
%         [d_hat, s_hat] = degrees(data, varargin) Estimate sparsity s_hat = 1./d_hat 
%         [d_hat, s_hat, sigma] = degrees(data, varargin) Return estimated kernel bandwidth
%
% Input: 
%       data           
%               N x D Data matrix. N rows are measurements, D columns are features.
%                   Accepts: 
%                       numeric
%   varargin: 
%       sigma      (default = 'std')
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
%       d_hat
%               N x 1 vector of the degree at each point in data of size N
%       s_hat
%               N x 1 vector of the sparsity at each point, s_hat=1./d_hat
%       sigma
%               N x 1 (adaptive) or scalar (constant) estimated kernel bandwidth
%  

[data, sigma, a,k,fac] = init(data,varargin{:});
N=size(data,2);

%construct kernel
[K,sigma] = gauss_kernel(data,data, 'sigma', sigma, 'a', a, 'k', k, 'fac', fac);
p = sum(K);

% Compute ouputs
d_hat=p*(N)/sum(p);

s_hat=1./d_hat;


end

function [data, sigma, a, k, fac] = init(data, varargin)
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
    default.sigma = 'std';
    default.k = 5;
    default.a = 2;
    default.fac = 1;
    %parser configuration
    persistent p
    if isempty(p)
        p = inputParser;
        addRequired(p, 'data', @isnumeric);
        addParameter(p, 'sigma', default.sigma,@check_sigma);
        addParameter(p, 'k', default.k, scalarPos);
        addParameter(p, 'a', default.a, scalarPos);
        addParameter(p,'fac', default.fac,scalarPos);
    end
    %parse
    parse(p, data, varargin{:})
    data = p.Results.data;
    sigma = p.Results.sigma;
    k = p.Results.k;
    a = p.Results.a;
    fac = p.Results.fac;
end

