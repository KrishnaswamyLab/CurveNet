close all; clear all;
addpath(genpath(pwd));
rng('shuffle');

quad_coeffs = [0.75, 1];
quad_scales = [16, 32];
is_saddle = [true, false];

intrinsic_dim = 2;
num_pts = 1000;

num_train = 100;
sampling_rad = 0.2;

triu_numel = intrinsic_dim*(intrinsic_dim+1)/2;

cnt = 1;

for cs = 1:numel(is_saddle)

    for sc = 1:numel(quad_scales)

        a = quad_coeffs(1);
        b = quad_coeffs(2);
        if is_saddle(cs) == true
            b = -b;
        end
    
        asc = a * quad_scales(sc);
        bsc = b * quad_scales(sc);

        Q = [asc 0; 0 bsc];

        test_X = zeros(num_train, num_pts+1, intrinsic_dim+1);
        test_y = zeros(num_train, triu_numel+1);

        for i = 1:num_train

            sample_pts = randsphere(1000, intrinsic_dim, sampling_rad);
            Z_pts = sample_pts*(Q*sample_pts');

            test_X(i,:,:) = [zeros(1,intrinsic_dim+1); [sample_pts diag(Z_pts)]];

            test_y(i,:) = [a b 0 quad_scales(sc)];

        end

        save(strcat('test_quadric_', string(cnt), '.mat'), 'test_X', 'test_y')

        cnt = cnt + 1;

    end
end