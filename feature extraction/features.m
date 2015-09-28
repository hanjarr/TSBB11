im_blue = imread('..\..\sydney\ortho_blue\0_0_0_tex.tif');
im_red = imread('..\..\sydney\ortho_red\0_0_0_tex.tif');
im_green = imread('..\..\sydney\ortho_green\0_0_0_tex.tif');
im_nir = imread('..\..\sydney\ortho_nir\0_0_0_tex.tif');

im = im_red.*0.2989;
im(:,:,2) = im_green(:,:,2).*0.5870;
im(:,:,3) = im_blue(:,:,3).*0.1140;

inImage = imgaussfilt(im(3500:4000, 1500:2000, :),8);
inImSize = size(inImage);

N=3; %NxN imSize

upperLimits = [floor(inImSize(1)/N)*N, floor(inImSize(2)/N)*N];
image = inImage(1:upperLimits(1),1:upperLimits(2));
imSize = size(image);

newImage = zeros(imSize(1)/N-1, imSize(2)/N-1, 4);

for i=1:N:imSize(1)-N
    for k=1:N:imSize(2)-N
        extImage = image(i:i+(N-1), k:k+(N-1));
        stats = imStats(extImage);
        
        newImage(((i-1)/N)+1, ((k-1)/N)+1, 1) = stats.Contrast;
        newImage(((i-1)/N)+1, ((k-1)/N)+1, 2) = stats.Correlation;
        newImage(((i-1)/N)+1, ((k-1)/N)+1, 3) = stats.Energy;
        newImage(((i-1)/N)+1, ((k-1)/N)+1, 4) = stats.Homogeneity;
        
        
    end
    counter = i/(imSize(1)-N)
end

figure(1)
subplot(1,5,1)
imshow(newImage(:,:,1))
title('Contrast')

subplot(1,5,2)
imshow(newImage(:,:,2))
title('Correlation')

subplot(1,5,3)
imshow(newImage(:,:,3))
title('Energy')

subplot(1,5,4)
imshow(newImage(:,:,4))
title('Homogeneity')

subplot(1,5,5)
imshow(im_blue(3500:4000,1500:2000, :))
title('Org_Image')



