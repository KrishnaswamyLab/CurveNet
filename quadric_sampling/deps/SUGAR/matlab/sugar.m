function [Y, out_labels, d_hat, s_hat, sigma, noise, npts, random_points, mgc_kernel, mgc_diffusion_operator] = sugar(data,varargin)
% SUGAR   Geometry-based Data Generation    https://arxiv.org/abs/1802.04927
%   Authors: Ofir Lindenbaum, Jay S. Stanley III.
%
% Usage:
%         Y = sugar(data, varargin) Generate Y using estimated manifold geometry over data
%         [~, out_labels] = sugar(data, varargin) Generated labels
%         [~, ~, d_hat] = sugar(data, varargin) Degree estimate of data
%         [~, ~, ~, s_hat] = sugar(data, varargin) Sparsity estimate of data
%         [~, ~, ~, ~, sigma] = sugar(data, varargin) Bandwidth used for degree computation
%         [~, ~, ~, ~, ~, noise] = sugar(data, varargin) Covariance estimate around each x_i in X.
%         [~, ~, ~, ~, ~, ~, npts] = sugar(data, varargin) Number of points estimate 
%         [~, ~, ~, ~, ~, ~, ~, random_points] = sugar(data, varargin) Random points before MGC correction
%         [~, ~, ~, ~, ~, ~, ~, ~, mgc_kernel] = sugar(data, varargin) MGC kernel over the new points through the data via s_hat
%         [~, ~, ~, ~, ~, ~, ~, ~, ~, mgc_diffusion_operator] = sugar(data, varargin) MGC diffusion operator used for MAGIC
% Input: 
%       data           
%               N x D Data matrix. N rows are measurements, D columns are features.
%                   Accepts: 
%                       numeric
%
%   varargin:
%       labels           (default = [])
%               N x 1 vector of classifier labels.
%                   Accepts numeric
%       noise_cov        (default = 'knn')
%               Bandwidth of Gaussian noise.
%                   Accepts:
%                       'knn' - (local covariance estimation for tuning noise)
%                       scalar - constant noise bandwidth
%`      noise_k      (default = 5)
%               Neighborhood size for covariance estimation
%                   Accepts:
%                       positive scalars
%       sparsity_idx (default = [])
%               Column indexes for sparsity estimation dimensions.
%                   [] : Estimate sparsity in all dimensions.
%       degree_sigma    (default = 'std')
%               Diffusion kernel bandwidth. 
%                   Accepts:    
%                        'std'- standard deviation of the distances
%                        'knn' - adaptive bandwidth,eg  kth neighbor distance
%                        'minmax'-  min-max on the distance matrix       
%                        'median'- the median of the distances      
%                         function handle - @(d) f(d) = scalar or N-length 
%                                   vector where d is an NxN distance matrix.    
%                         scalar - pre-computed bandwidth
%
%       degree_k         (default = 5)
%               k-Nearest neighbor distance to use if degree_sigma = 'knn'
%               Accepts:
%                       positive scalars
%       degree_a         (default = 2)
%               Alpha-kernel decay parameter for degree computation. 2 is Gaussian kernel.
%               Accepts:
%                       positive scalars
%       degree_fac         (default = 1)
%               Rescale mgc kernel bandwidth
%                   Accepts:
%                       positive scalars
%       M               (default = 0)
%               Number of points to generate.  Can affect strength of density
%               equalization.
%                   Accepts: 
%                        positive scalars 
%                        If (M && equalize) then density equalization will be
%                        scaled by M.  M < N will negatively impact density
%                        equalization, M << N is not recommended and M <<< N may fail.
%                        If (~M && equalize) then density equalization will not be
%                        scaled
%                        If (M && ~equalize) then approximately M points will be
%                        generated according to a constant difference of the
%                        max density
%                        If (~M && ~equalize) then M = approx. N points will be
%                        generated.
%
%       equalize        (default = false)
%               Density equalization.  Can be affected by M.
%                   Accepts: 
%                       logical / scalar
%
%       mgc_magic       (default = 1)
%               Apply MGC MAGIC to diffuse new points.
%               mgc_magic = 0 disables this function.
%               mgc_magic > 0 applies mgc_magic steps of diffusion. 
%                   Accepts:
%                       Scalar
%       mgc_sigma    (default = 'knn')
%               Diffusion kernel bandwidth. 
%                   Accepts:    
%                        'std'- standard deviation of the distances
%                        'knn' - adaptive bandwidth,eg  kth neighbor distance
%                        'minmax'-  min-max on the distance matrix       
%                        'median'- the median of the distances      
%                         function handle - @(d) f(d) = scalar or N-length 
%                                   vector where d is an NxN distance matrix.    
%                         scalar - pre-computed bandwidth
%       mgc_a               (default = 2)
%               Alpha-kernel decay parameter for MGC. 2 is Gaussian kernel.
%                   Accepts:
%                        positive scalars
%       mgc_k         (default = 5)
%               k-Nearest neighbor distance to use if mgc_sigma = 'knn'
%                   Accepts:
%                       positive scalars
%       mgc_fac         (default = 1)
%               Rescale mgc kernel bandwidth
%                   Accepts:
%                       positive scalars
%       magic_rescale   (default = 1)
%               Rescale new points after magic.
%                   Accepts:
%                       positive scalars
%       suppress (default = false)
%               Enable/disable point generation errors
%                   Accepts: 
%                       logical / scalar
% Output:
%      Y 
%               M x D New points generated by SUGAR
%      out_labels
%               M x 1 New labels corresponding to Y.
%      d_hat
%               N x 1 Degree estimate of data
%      s_hat
%               N x 1 Sparsity estimate of data
%      sigma
%               N x 1 or Scalar sigma used for d_hat computation
%      noise
%               Scalar - Constant noise bandwidth
%               N x D x D cell of local covariance matrices on X
%      npts
%               N x 1 Number of points generated for each point in X
%      random_points
%               M x D Random points generated around X
%      mgc_kernel
%               M x M kernel matrix of Y through data over s_hat
%      mgc_diffusion_operator
%               M x M Makov matrix of mgc_kernel
%
  
disp('Initializing SUGAR')
[data, params] = init(data, varargin{:});


%Estimate the degree
disp('Obtaining Degree Estimate')
if isempty(params.sparsity_idx)
    params.sparsity_idx = [1:size(data,2)];
end
[d_hat, s_hat, sigma] = degrees(data(:,params.sparsity_idx), 'sigma', params.degree_sigma, 'k', params.degree_k, 'a', params.degree_a, 'fac', params.degree_fac);
params.degree_sigma = sigma;

%If no parameter for noise covariance is given then estimate local covariance
if strcmp(params.noise_cov, 'knn')
    disp('Local Covariance estimation')
    [noise] = local_covariance(data,'k',params.noise_k);
    params.noise_cov = noise;
end

%Estimate number of points to generate around each original point
disp('Estimating number of points to generate')
[npts] = numpts(d_hat, 'noise_cov', params.noise_cov, 'kernel_sigma', params.degree_sigma, 'dim', params.dim,'M', params.M, 'equalize', params.equalize,'suppress', params.suppress);
%Generate points 
disp('Generating points')
[random_points, out_labels] = generate(data,npts,params.noise_cov,params.labels);

%Compute and apply mgc magic
if params.mgc_magic > 0
    disp('Diffusing points via MGC MAGIC')
    [Y, mgc_kernel, mgc_diffusion_operator] = mgc_magic(data, random_points,s_hat,'sigma', params.mgc_sigma, ...
    'a',params.mgc_a,'k',params.mgc_k, 't', params.mgc_magic, ...
    'fac', params.mgc_fac, 'magic_rescale', params.magic_rescale );
else
    Y = random_points;
    mgc_kernel = [];
    mgc_diffusion_operator = [];
end



end

function [data, params] = init(data, varargin)
% initialization & parameter parsing
% SUGAR DEFAULTS
    default.noise_cov = 'knn';
    default.noise_k = 5;
    
    default.degree_sigma = 'std';
    default.degree_a = 2;
    default.degree_k = 5;
    default.degree_fac = 1;
    
    default.M = 0; % we will check this later and change it to be based on the data size N
    default.equalize = false;
    default.labels = [];
    default.sparsity_idx =[];
    default.suppress = false;
    
    default.mgc_magic = 1;
    default.mgc_a = 2;
    default.mgc_k = 5;
    default.mgc_sigma = 'knn';
    default.mgc_fac = 1;
    default.magic_rescale = 1;
% type checking
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

    function tf = check_noise(passed)
    % check gaussian noise noise
        if isscalar(passed) || strcmp(passed, 'knn')
            tf = true;
        else 
            tf = false;
        end
    end

    scalarPos = @(x) isscalar(x) && (x>0);
    persistent p
    if isempty(p)
        %configure parser
        p = inputParser;
        addRequired(p, 'data', @isnumeric);
        addParameter(p, 'noise_cov', default.noise_cov, @check_noise);
        addParameter(p, 'noise_k', default.noise_k, scalarPos);
        
        addParameter(p, 'degree_sigma', default.degree_sigma,@check_sigma);
        addParameter(p, 'degree_k', default.degree_k, scalarPos);
        addParameter(p, 'degree_a', default.degree_a, scalarPos);
        addParameter(p, 'degree_fac', default.mgc_fac, @isscalar);

        addParameter(p, 'sparsity_idx', default.sparsity_idx, @isnumeric);
        addParameter(p, 'M', default.M, scalarPos);
        addParameter(p, 'equalize', default.equalize, @isscalar);
        addParameter(p, 'labels', default.labels, @isnumeric);
        addParameter(p, 'suppress', default.suppress, @isscalar);
        
        addParameter(p, 'mgc_magic', default.mgc_magic, @isscalar);
        addParameter(p, 'mgc_a', default.mgc_a, @isscalar);
        addParameter(p, 'mgc_sigma', default.mgc_sigma, @check_sigma);
        addParameter(p, 'mgc_k', default.mgc_k, @isscalar);
        addParameter(p, 'mgc_fac', default.mgc_fac, @isscalar);
        addParameter(p, 'magic_rescale', default.magic_rescale, @isscalar);

    end
    % parse
    parse(p,data, varargin{:})
    
    data = p.Results.data;
    params = p.Results;
    params = rmfield(params,'data');
    params.dim = size(data, 2);
    
end
