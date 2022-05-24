function npts = numpts(degree,varargin)
%NUMPTS Compute the number of new points to generate around each point in a
%       dataset     https://arxiv.org/abs/1802.04927
% Authors: Ofir Lindenbaum, Jay S. Stanley III.
%
% Usage:
%         npts = numpts(degree, varargin) Generate npts, the estimate of
%         the number of points generate according to degree.
%
% Input: 
%       degree
%               Degree estimate of the N x D data
%                   Accepts:
%                       N x 1 numeric
%   varargin:
%       noise_cov       (default = 1)
%               Noise bandwidth used for downstream data generation
%                   Accepts:
%                       Scalar - uniform Gaussian noise variance
%                       N x 1 cell - contains D x D local covariance
%                       matrices for Gaussian generated noise
%
%       kernel_sigma        (default = 1)
%               Degree estimate bandwidth
%                   Accepts:
%                       N x 1 numeric - adaptive bandwidth 
%                       Scalar - uniform bandwidth
%
%                  
%       dim             (default = D if available from noise_cov, else required)       
%               Generated noise dimension
%                   Accepts:
%                       Scalar
%
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
%       suppress (default = false)
%               Enable/disable point generation errors
%
% Output
%       npts the number of points to generate at each point
%     
[degree, noise_cov, kernel_sigma, dim, M, equalize, suppress] = init(degree, varargin{:});
N = length(degree);
Const=max(degree);
NumberEstimate = zeros(N, 1);
if equalize
        if isscalar(noise_cov)
                disp("Density equalization according to scalar noise covariance...")
                NumberEstimate=(Const-degree)*((kernel_sigma^2+noise_cov^2)/((2)*noise_cov^2))^(dim/2);
        else
            disp("Density equalization according to local noise covariance...")

            for i=1:N
                if ~isscalar(kernel_sigma)
                    sig = kernel_sigma(i);
                else
                    sig = kernel_sigma;
                end
                NumberEstimate(i)=(Const-degree(i))*det(eye(size(noise_cov{i}))+noise_cov{i}./(2*sig.^2)).^(0.5);

            end
            
        end
        if logical(M)
            disp("Applying total generation constraint M.")
            number_save = NumberEstimate;
            number_sum = sum(NumberEstimate);
            if M/number_sum < 1e-1
                warning(['Supplied M is ' num2str((M/number_sum)*100) '% of equalized total. ', 'Output will not reflect equalization. Increased M is suggested.'])
            end
            NumberEstimate = NumberEstimate*M/(sum(NumberEstimate)+1e-17);
            npts = floor(NumberEstimate);
        else
            npts = floor(NumberEstimate);
        end
else
    disp("Generating without density equalization")
    if ~logical(M)
        disp("No M supplied, M = N.")
        M = N;
    end
    NumberEstimate=(Const-degree);
    NumberEstimate = NumberEstimate*M/(sum(NumberEstimate)+1e-17);
    
    npts = floor(NumberEstimate);
end
if ~suppress
    errfunc = @(x) error(x);
else
    errfunc = @(x) disp(x);
end
if sum(npts)==0
    errfunc('Point generation estimate < 0 , either provide/increase M or decrease noise_cov');
    npts = ones(N,1);
elseif sum(npts)>10^4
    errfunc('Point generation > 1e4, either provide/decrease M or increase noise_cov');
end
end

function [degree, noise_cov, kernel_sigma, dim, M, equalize, suppress] = init(degree, varargin)
    % helpers
    scalarPos = @(x) isscalar(x) && (x>0);
    check_noise = @(x) isscalar(x) || (iscell(x) && ~isempty(x) && size(x{1},1) == size(x{1},2));
    check_vec = @(x) isnumeric(x) && min(size(x))==1;

    % defaults
    default.noise_cov = 1;
    default.kernel_sigma = 1;
    default.dim = [];
    default.M = 0;
    default.equalize = false;
    default.suppress = true;
    persistent p
    
    if isempty(p)
        % configure parser
        p = inputParser;
        addRequired(p, 'degree', check_vec);
        addParameter(p, 'noise_cov', default.noise_cov, check_noise);
        addParameter(p, 'kernel_sigma', default.kernel_sigma, check_vec);
        addParameter(p, 'dim', default.dim, scalarPos);
        addParameter(p, 'M', default.M, @isscalar);
        addParameter(p, 'equalize', default.equalize, @isscalar);
        addParameter(p, 'suppress', default.suppress, @isscalar);

    end
    % parse
    parse(p, degree, varargin{:});
    degree = p.Results.degree;
    noise_cov = p.Results.noise_cov;
    kernel_sigma = p.Results.kernel_sigma;
    dim = p.Results.dim;
    M = p.Results.M;
    equalize = p.Results.equalize;
    suppress = p.Results.suppress;
    if isempty(dim) % check if dimension supplied
        if iscell(noise_cov) % we can get the dimension from local covariance matrices
            dim = size(noise_cov{1},1);
        else
            error("'dim' is required if 'noise_cov' is not a cell")
        end
    end
end
function fp = feature_scale(x)
    fp = (x-min(x))./(max(x)-min(x));
end