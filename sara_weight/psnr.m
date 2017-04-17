function out = psnr( img1, img2 )
% PSNR
% The pixel value is assumed to be from 0~255

if (size(img1, 1) ~= size(img2, 1)) || (size(img1, 2) ~= size(img2, 2))
    msg = 'Images must have the same size!';
    error(msg);
end

x1 = double(img1(:));
x2 = double(img2(:));
mse = @(z1, z2) norm(z1 - z2)^2 / numel(z1);

out = 10*log10( 255^2 / (mse(x1, x2) + eps) );


end

