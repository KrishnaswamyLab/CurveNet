close all; clear all;

rng(22519)

% params to vary
intrinsic_dim = 2;
num_pts = 1000;

% fixed params
num_train = 10000;
max_scale = 64;
sampling_rad = 0.2;

triu_numel = intrinsic_dim*(intrinsic_dim+1)/2;
train_X = zeros(num_train, num_pts+1, intrinsic_dim+1);
train_y = zeros(num_train, triu_numel+1);

for i = 1:num_train

    A = -1 + 2*rand(intrinsic_dim);
    A = (A + A')/2;
    sc = max_scale * rand();

    D = diag(A);
    coeff = [D.', squareform((A-diag(D)).')];
    train_y(i,:) = [coeff sc];

    Q = sc * A;

    sample_pts = randsphere(1000, intrinsic_dim, sampling_rad);
    Z_pts = sample_pts*(Q*sample_pts');

    train_X(i,:,:) = [zeros(1,intrinsic_dim+1); [sample_pts diag(Z_pts)]];

end

save(strcat('random_quadrics_dim_', string(intrinsic_dim), '.mat'), 'train_X', 'train_y')