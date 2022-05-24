function [data_imputed, diffusion_operator] = magic(data,kernel,t, rescale)
% MAGIC   Markov Affinity based Gaussian Imputation of Cells 
%   See
%       https://www.biorxiv.org/content/early/2017/02/25/111591
%       https://arxiv.org/abs/1802.04927
%   Implementation Authors: Ofir Lindenbaum, Jay S. Stanley III.
%   Publication Authors: David van Dijk et al. 
% Usage:
%         data_imputed = magic(data, kernel, t, rescale) Impute data using kernel for t time steps with rescaling
%         [~, diffusion_operator] = magic(data, kernel, t, rescale) Diffusion operator used for data_imputed
% Input: 
%       data           
%               N x D Data matrix. N rows are measurements, D columns are features.
%                   Accepts: 
%                       numeric
%       kernel      
%               N x N Kernel or affinity matrix. 
%                   Accepts:
%                       numeric
%       t       (default = 1)
%               Scalar. Time steps to apply MAGIC for.  Controls low pass filter cutoff.
%       rescale (default = true)
%               Rescale 95th percentile of imputed data to match original data.
% Output:
%      data_imputed 
%               M x D Data points imputed via MAGIC using kernel.
%      diffusion_operator
%               M x M Markov matrix built from kernel.
%

if nargin == 2
    warning('nargin = 2, t set to 1, rescale set to true')
    t = 1;
    rescale = true;
elseif nargin == 3
    warning('nargin = 3, rescale set to true')
    rescale = true;
end

if size(data, 1) ~= size(kernel,1) % check data compatibility with kernel
    if size(data, 2) ~= size(kernel,1)
        error('Data does not match kernel size')
    else
        data = data';
    end
end
% build diffusion operator
diffusion_degrees = diag(sum(kernel,2))^-1;
diffusion_degrees(isinf(diffusion_degrees)) = 0;
diffusion_operator = diffusion_degrees*kernel;

clear diffusion_degrees kernel % memory (do we need this?)

data_imputed = data;

for i = 1:t

data_imputed = diffusion_operator * data_imputed;

if rescale
    data_imputed = (data_imputed) .* (prctile(data,95)./prctile(data_imputed,95));
end
end
end
