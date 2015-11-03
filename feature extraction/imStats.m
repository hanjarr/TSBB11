function [stats] = imStats(subImage) 
% Pair i 4 vinklar
[glcm0] = graycomatrix(subImage,'offset', [0 1], 'NumLevels',8, 'G', []);
[glcm135] = graycomatrix(subImage,'offset', [-1 -1], 'NumLevels', 8, 'G', []);
[glcm45] = graycomatrix(subImage,'offset', [-1 1], 'NumLevels', 8, 'G', []);
[glcm90] = graycomatrix(subImage,'offset', [-1 0], 'NumLevels', 8, 'G', []);

glcm0_norm = glcm0./(sum(sum(glcm0)));
glcm45_norm = glcm45./(sum(sum(glcm45)));
glcm90_norm = glcm90./(sum(sum(glcm90)));
glcm135_norm = glcm135./(sum(sum(glcm135)));

glcm_sum = round(1000*(glcm0_norm+glcm45_norm+glcm90_norm+glcm135_norm));

stats = graycoprops(glcm_sum);
end