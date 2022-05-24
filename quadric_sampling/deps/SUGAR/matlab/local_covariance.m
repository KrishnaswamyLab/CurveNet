function [local_cov] = local_covariance(data,varargin)
% LOCAL_COVARIANCE Compute k-nn neighborhood covariance
% Authors: Ofir Lindenbaum, Jay S. Stanley III.     https://arxiv.org/abs/1802.04927
% Usage:
%         [local_cov] = local_covariance(data) Estimate local covariance
%         around each point in the data 
%         [local_cov] = local_covariance(data, 'k', k) Estimate local covariance
%         around each point in the data with neighborhood size k
%
% Input: 
%       data           
%               N x D Data matrix. N rows are measurements, D columns are features.
%                   Accepts: 
%                       numeric
%   varargin:
%       k           (default = 5)
%               number of nearest neighbors for local covariance
%                   Accepts:
%                       positive scalars
%
% Output: 
%       local_cov
%               N-length cell of local covariance matrices of the Gaussian
%                   generated noise. 
%     

% parse helper
scalarPos = @(x) isscalar(x) && (x>0);

% defaults
default.k = 5;

% parse config
persistent p
if isempty(p)
    p = inputParser;
    addRequired(p, 'data', @isnumeric);
    addParameter(p, 'k', default.k, scalarPos);
end
% parse
parse(p,data,varargin{:});
k = p.Results.k;
data = p.Results.data;


% initialize

% compute covariances
for i=1:size(data,1)
        IDX = knnsearch(data,data(i,:),'K',k); %find neighborhood
        local_cov{i}=cov(data(IDX,:)); %take covariance
end

end
