function extra(file_name1)

filename='flowers.tiff';

% dimensione of statistics
Nb = [2, 8];
% number of cumulated bloksl
Ns = 1;
bayer = [0, 1;
         1, 0];
     
     
     im_true = imread(filename);
     im = imread(file_name1);
     demo(imread(file_name1));
       
     for j = 1:1
        
        [map, stat] = Process(im, bayer, Nb(j),Ns);

        [h w] = size(map);

        %    NaN and Inf management

        stat(isnan(stat)) = 1;
        data = log(stat(:)); 
        data = data(not(isinf(data)|isnan(data)));
        % square root rule for bins
        n_bins = round(sqrt(length(data)));

        % plot result
        figure
        subplot(2,2,1), imshow(im_true), title('Not tampered image');
        subplot(2,2,2), imshow(im), title('Manipulated image');
        subplot(2,2,3), imagesc(map), colormap('gray'),axis equal, axis([1 w 1 h]), title(['Posability map']);
        subplot(2,2,4), hist(data, n_bins), title('Histogram of the proposed feature');
   
     end
origImg = im_true;
distImg = im;

%If the input image is rgb, convert it to gray image
noOfDim = ndims(origImg);
if(noOfDim == 3)
    origImg = rgb2gray(origImg);
end

noOfDim = ndims(distImg);
if(noOfDim == 3)
    distImg = rgb2gray(distImg);
end

%Size Validation
origSiz = size(origImg);
distSiz = size(distImg);
sizErr = isequal(origSiz, distSiz);
if(sizErr == 0)
    disp('Error: Original Image & Distorted Image should be of same dimensions');
    return;
end

%Mean Square Error 
MSE = MeanSquareError(origImg, distImg);
disp('Mean Square Error = ');
disp(MSE);

%Peak Signal to Noise Ratio 
PSNR = PeakSignaltoNoiseRatio(origImg, distImg);
disp('Peak Signal to Noise Ratio = ');
disp(PSNR);

%Normalized Cross-Correlation 
NK = NormalizedCrossCorrelation(origImg, distImg);
disp('MNormalized Cross-Correlation  = ');
disp(NK);

%Average Difference 
AD = AverageDifference(origImg, distImg);
disp('Average Difference  = ');
disp(AD);

%Structural Content 
SC = StructuralContent(origImg, distImg);
disp('Structural Content  = ');
disp(SC);

%Maximum Difference 
MD = MaximumDifference(origImg, distImg);
disp('Maximum Difference = ');
disp(MD);

%Normalized Absolute Error
NAE = NormalizedAbsoluteError(origImg, distImg);
disp('Normalized Absolute Error = ');
disp(NAE);

end