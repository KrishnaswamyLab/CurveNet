function [new_data, mgc_kernel, mgc_diffusion_operator] = mgc_magic(X,Y,s_hat,varargin)
% MGC_MAGIC   Measure-based Gaussian Correlation Kernel w/ MAGIC  https://arxiv.org/abs/1802.04927
%   Authors: Ofir Lindenbaum, Jay S. Stanley III.
%
% Usage:
%         new_data = mgc_magic(X,Y, s_hat, varargin) Impute Y using an MGC kernel between X via the measure S_hat
% Input: 
%       X           
%               N x D Data matrix. N rows are measurements, D columns are features.
%                   Accepts: 
%                       numeric
%       Y       M x D Data matrix.  M rows are measurements, D columns are features.
%                   Accepts:
%                       numeric
%       s_hat   M x 1 measure vector. M entries correspond to measures on the rows of Y.
%                   Accepts:
%                       M x 1 numeric
%
%   varargin:
%       sigma    (default = 'knn')
%               MGC kernel bandwidth. 
%                   Accepts:    
%                        'std'- standard deviation of the distances
%                        'knn' - adaptive bandwidth,eg  kth neighbor distance
%                        'minmax'-  min-max on the distance matrix       
%                        'median'- the median of the distances      
%                         function handle - @(d) f(d) = scalar or N-length 
%                                   vector where d is an NxN distance matrix.    
%                         scalar - pre-computed bandwidth
%
%       k         (default = 5)
%               k-Nearest neighbor distance to use if sigma = 'knn'
%               Accepts:
%                       positive scalars
%       a         (default = 2)
%               Alpha-kernel decay parameter for kernel. 2 is Gaussian kernel.
%               Accepts:
%                       positive scalars
%       fac         (default = 1)
%               Rescale mgc kernel bandwidth
%                   Accepts:
%                       positive scalars
%       t       (default = 1)
%               Apply MGC MAGIC to diffuse new points.
%               mgc_magic = 0 disables this function.
%               mgc_magic > 0 applies mgc_magic steps of diffusion. 
%                   Accepts:
%                       Scalar
%       magic_rescale   (default = 1)
%               Rescale new points after magic.
%                   Accepts:
%                       positive scalars
% Output:
%      new_data 
%               M x D New points corrected using MGC MAGIC through X.
%      mgc_kernel
%               M x M MGC kernel built over Y through X via s_hat
%      mgc_diffusion_operator
%               M x M row stochastic MGC markov matrix / diffusion operator

[X,Y, s_hat, sigma,a,k,fac,t,magic_rescale] = init(X,Y,s_hat,varargin{:});
if t == 0
    new_data = y;
    warning('mgc_magic was passed t=0, no mgc_magic was performed')
    return
end

new_to_old = gauss_kernel(Y,X,'sigma', sigma,'a', a, 'k', k, 'fac', fac);
old_to_new = gauss_kernel(X,Y,'sigma', sigma, 'a', a, 'k', k, 'fac', fac);

new_to_old_sparsity=bsxfun(@times,new_to_old,s_hat);

mgc_kernel = new_to_old_sparsity*old_to_new; 

mgc_kernel = (mgc_kernel+mgc_kernel')./2;
[new_data, mgc_diffusion_operator] = magic(Y, mgc_kernel, t, magic_rescale);



end

function [X, Y, s_hat, sigma, a, k, fac, t, magic_rescale] = init(data, varargin)
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
    default.t = 1;
    default.magic_rescale = 1;

    %parser configuration
    persistent p
    if isempty(p)
        p = inputParser;
        addRequired(p, 'X', @isnumeric);
        addRequired(p, 'Y', @isnumeric);
        addRequired(p, 's_hat', @isnumeric);
        addParameter(p, 'sigma', default.sigma,@check_sigma);
        addParameter(p, 'k', default.k, scalarPos);
        addParameter(p, 'a', default.a, scalarPos);
        addParameter(p,'fac', default.fac,scalarPos);
        addParameter(p,'t', default.t,scalarPos);
        addParameter(p,'magic_rescale', default.magic_rescale, @isscalar);

    end
    %parse
    parse(p, data, varargin{:})
    X = p.Results.X;
    Y = p.Results.Y;
    s_hat = p.Results.s_hat;
    sigma = p.Results.sigma;
    k = p.Results.k;
    a = p.Results.a;
    fac = p.Results.fac;
    t = p.Results.t;
    magic_rescale = p.Results.magic_rescale;

end
