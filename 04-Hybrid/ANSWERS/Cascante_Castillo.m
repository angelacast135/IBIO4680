%% Hybrid Images
clear all, close all

mkdir('data')
fullURL =['https://www.dropbox.com/sh/w4cqjdkdzsor3df/AACtBA3jPWwQDu4adTh864yDa?dl=1'];
filename = 'images.zip';
a_url=urlwrite(fullURL,filename);
a_zip=unzip(filename, 'data');

fils = dir(fullfile('./data','*.jpg'));
ims = zeros(256,256,3,length(fils));
ims_lp = zeros(256,256,3,length(fils));
ims_hp = zeros(256,256,3,length(fils));

ims_hyb = zeros(256,256,3,length(fils),10);

% cut frequency of LP filter
fc1 = 0.2;
% cut frequency of HP filter
fc2 =  1;

ker1 = fspecial('gaussian',[19,19],1);
ker2 = fspecial('gaussian',[21,21],9);

for i=1:length(fils)
    ims(:,:,:,i)=imresize((imread( fullfile('./data',fils(i).name))),[256,256]);
    
    %low pass filter
    ims_lp(:,:,:,i) =  im2double((imfilter(uint8(ims(:,:,:,i)),ker1,'same')));
    
    %HP filter
    ims_hp(:,:,:,i) = im2double(uint8(ims(:,:,:,i)))-im2double(imfilter(uint8(ims(:,:,:,i)),ker2,'same'));
    ims_hp(:,:,:,i) = (ims_hp(:,:,:,i)-min(min(ims_hp(:,:,:,i))))./(max(max(ims_hp(:,:,:,i)))-min(min(ims_hp(:,:,:,i))));

    for j=real(1:length(fils))
        ims_hyb(:,:,:,i,j) =  ims_hp(:,:,:,i)+ims_lp(:,:,:,j);
    end
end

i = 2; j=1;
subplot(221),imshow((ims_lp(:,:,:,i)),[]), title('Low-pass of Johnatan')
subplot(222),imshow((ims_hp(:,:,:,i)),[]), title('High-pass of Johnatan')

subplot(223),imshow((ims_hyb(:,:,:,i,j)),[]), title('Hybrid Image')
subplot(224),imshow((ims_lp(:,:,:,j)),[]), title('Low-pass of Juanita')

figure 
output=vis_hybrid_image(ims_hyb(:,:,:,i,j));
imshow((output),[])
title('Pyramid')
%% Pyramid blending
clear all, close all
%number of levels in the pyramid
level = 6;

%gaussian filter for sampling
ker1 = fspecial('gauss',[9,9],5); 

% create the gaussian and laplacian pyramids for each image.
[ pyr_gauss1, pyr_lap1 ] = create_pyramids( imread('./data/Juanita.jpg'), level ,ker1);
[ pyr_gauss2, pyr_lap2 ] = create_pyramids( imread('./data/Sanchez.jpg'), level ,ker1);

%number of levels to 
level_2_cat = level-1;
%save each blending image
pyr_blend = cell(1,level_2_cat);

% cat the last levels of the pyramid
pyr_blend{1} =  [pyr_gauss1{level_2_cat}(:,1:size(pyr_gauss1{level_2_cat},2)/2-1,:), pyr_gauss2{5}(:,size(pyr_gauss1{level_2_cat},2)/2:end,:)];

for i=1:level_2_cat-1
    %Gaussian Pyramid up. and add then Add the corresponding Laplacian Pyramid.
    pyr_blend{i+1} = im2double(imresize(imfilter((pyr_blend{i}),ker1,'same'),2, 'bilinear' ))...
        +im2double([pyr_lap1{level_2_cat-i}(:,1:size(pyr_lap1{level_2_cat-i},2)/2-1,:), pyr_lap2{level_2_cat-i}(:,size(pyr_lap2{level_2_cat-i},2)/2:end,:)]);
    pyr_blend{1,i+1} = im2double(pyr_blend{1,i+1}-min(min(pyr_blend{1,i+1})))./( max(max(pyr_blend{1,i+1}))-min(min(pyr_blend{1,i+1})) );
end
%show the blended image
imshow(pyr_blend{end})