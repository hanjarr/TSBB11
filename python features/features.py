import cv2
# im_blue = imread('..\..\sydney\ortho_blue\0_0_0_tex.tif');
# im_red = imread('..\..\sydney\ortho_red\0_0_0_tex.tif');
# im_green = imread('..\..\sydney\ortho_green\0_0_0_tex.tif');
# im_nir = imread('..\..\sydney\ortho_nir\0_0_0_tex.tif');

im_blue = cv2.imread("..\..\sydney\ortho_blue\0_0_0_tex.tif")
im_red = cv2.imread("..\..\sydney\ortho_red\0_0_0_tex.tif")
im_green = cv2.imread("..\..\sydney\ortho_green\0_0_0_tex.tif")
im_nir = cv2.imread("..\..\sydney\ortho_nir\0_0_0_tex.tif")

# im = im_red.*0.2989;
# im(:,:,2) = im_green(:,:,2).*0.5870;
# im(:,:,3) = im_blue(:,:,3).*0.1140;

# impart = im(4000:4500, 2000:2500, :);

# inImage = imgaussfilt(impart, 8);
# inImSize = size(inImage);

# N=4; %NxN imSize

# upperLimits = [floor(inImSize(1)/N)*N, floor(inImSize(2)/N)*N];
# image = inImage(1:upperLimits(1),1:upperLimits(2));
# imSize = size(image);

# newImage = zeros(imSize(1)/N-1, imSize(2)/N-1, 4);

# for i=1:N:imSize(1)-N
    # for k=1:N:imSize(2)-N
        # extImage = image(i:i+(N-1), k:k+(N-1));
        # stats = imStats(extImage);
        
        # newImage(((i-1)/N)+1, ((k-1)/N)+1, 1) = stats.Contrast;
        # newImage(((i-1)/N)+1, ((k-1)/N)+1, 2) = stats.Correlation;
        # newImage(((i-1)/N)+1, ((k-1)/N)+1, 3) = stats.Energy;
        # newImage(((i-1)/N)+1, ((k-1)/N)+1, 4) = stats.Homogeneity;
        
        
    # end
    # counter = i/(imSize(1)-N)
# end

# newImage(:,:,1) = newImage(:,:,1)/max(max(newImage(:,:,1)));
# newImage(:,:,2) = newImage(:,:,2)/max(max(newImage(:,:,2)));
# newImage(:,:,3) = newImage(:,:,3)/max(max(newImage(:,:,3)));
# newImage(:,:,4) = newImage(:,:,4)/max(max(newImage(:,:,4)));

print "Hello world"


