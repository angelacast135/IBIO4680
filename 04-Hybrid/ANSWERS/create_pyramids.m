function [ pyr_gauss, pyr_lap ] = create_pyramids( img, level ,ker)

pyr_gauss1 = cell(1,level+1);
pyr_lap1 = cell(1,level);
% resize the original image in powers of 2
pyr_gauss1{1} = imresize(im2double(img),[1024,512]);

for i=1:level
    pyr_gauss1{1,i+1} = im2double(imresize(imfilter((pyr_gauss1{i}),ker,'same'),0.5, 'bilinear' ));
    
    %pyr_gauss1{1,i+1} =  (pyr_gauss1{1,i+1}-min(min(pyr_gauss1{1,i+1})))./( max(max(pyr_gauss1{1,i+1}))-min(min(pyr_gauss1{1,i+1})) );
    pyr_lap1{1,i} = pyr_gauss1{i}-im2double(imresize((pyr_gauss1{i+1}),2,'bilinear'));
    pyr_lap1{1,i} = (pyr_lap1{1,i}-min(min(pyr_lap1{1,i})))./(max(max(pyr_lap1{1,i}))-min(min(pyr_lap1{1,i})) );
end
pyr_gauss=pyr_gauss1;
pyr_lap=pyr_lap1;
end