im_blue = imread('..\..\sydney\ortho_blue\0_0_0_tex.tif');
im_red = imread('..\..\sydney\ortho_red\0_0_0_tex.tif');
im_green = imread('..\..\sydney\ortho_green\0_0_0_tex.tif');
im_nir = imread('..\..\sydney\ortho_nir\0_0_0_tex.tif');

im = im_red.*0.2989;
im(:,:,2) = im_green(:,:,2).*0.5870;
im(:,:,3) = im_blue(:,:,3).*0.1140;

impart = im(4000:4500, 2000:2500, :);

inImage = imgaussfilt(impart, 8);
inImSize = size(inImage);

N=4; %NxN imSize

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

newImage(:,:,1) = newImage(:,:,1)/max(max(newImage(:,:,1)));
newImage(:,:,2) = newImage(:,:,2)/max(max(newImage(:,:,2)));
newImage(:,:,3) = newImage(:,:,3)/max(max(newImage(:,:,3)));
newImage(:,:,4) = newImage(:,:,4)/max(max(newImage(:,:,4)));

figure(1)
subplot(2,5,1)
imshow(newImage(:,:,1))
title('Contrast')

subplot(2,5,2)
imshow(newImage(:,:,2))
title('Correlation')

subplot(2,5,3)
imshow(newImage(:,:,3))
title('Energy')

subplot(2,5,4)
imshow(newImage(:,:,4))
title('Homogeneity')

subplot(2,5,5)
imshow(impart)
title('inImage')

%%Rad 2
lowpass = newImage;
lowpass(:,:,1) = imgaussfilt(lowpass(:,:,1),0.5);
lowpass(:,:,1) = im2bw(lowpass(:,:,1), 0.5);

lowpass(:,:,2) = imgaussfilt(lowpass(:,:,2),0.5);
lowpass(:,:,2) = im2bw(lowpass(:,:,2), 0.5);

lowpass(:,:,3) = imgaussfilt(lowpass(:,:,3),1);
lowpass(:,:,3) = im2bw(lowpass(:,:,3), 0.3);

lowpass(:,:,4) = imgaussfilt(lowpass(:,:,4),0.5);
lowpass(:,:,4) = im2bw(lowpass(:,:,4), 0.8);

subplot(2,5,6)
imshow(lowpass(:,:,1))
title('Low Contrast')

subplot(2,5,7)
imshow(lowpass(:,:,2))
title('Low Corr')

subplot(2,5,8)
imshow(lowpass(:,:,3))
title('Low Energy')

subplot(2,5,9)
imshow(lowpass(:,:,4))
title('Low Homo')


