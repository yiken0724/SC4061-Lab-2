% Name: Choo Yi Ken
% Matric No.: U2240710B
% Title: SC4061 Lab 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.1.a - Otsu's Global Thresholding

imageFiles = {'document01.bmp', 'document02.bmp', 'document03.bmp', 'document04.bmp'};
gtFiles = {'document01-GT.tiff', 'document02-GT.tiff', 'document03-GT.tiff', 'document04-GT.tiff'};

% Entry point for processing image with a for loop
for i = 1:length(imageFiles)
    processImageWithOtsu(imageFiles{i}, gtFiles{i});
end

function processImageWithOtsu(imgFile, gtFile)
    [img, gt] = loadAndPreprocessImages(imgFile, gtFile);
    [binaryImg, otsuLevel] = applyOtsuThresholding(img);
    [diffSum, accuracy, binaryImg, diffImg] = compareWithGroundTruthOtsu(binaryImg, gt, imgFile);  
    displayOtsuResults(imgFile, otsuLevel, diffSum, accuracy);
    visualizeOtsuThreshold(img, otsuLevel);
    visualizeOtsuSegmentation(img, binaryImg, gt, diffImg, diffSum, otsuLevel);
end

function [img, gt] = loadAndPreprocessImages(imgFile, gtFile)
    img = imread(imgFile);
    gt = imread(gtFile);
    
    % Convert to grayscale
    if size(img, 3) == 3, img = rgb2gray(img); end
    if size(gt, 3) == 3, gt = rgb2gray(gt); end
    
    % Ensure ground truth is binary
    if ~islogical(gt), gt = imbinarize(gt); end
end

function [binaryImg, otsuLevel] = applyOtsuThresholding(img)
    otsuLevel = graythresh(img);
    binaryImg = imbinarize(img, otsuLevel);
end

function [diffSum, accuracy, binaryImg, diffImg] = compareWithGroundTruthOtsu(binaryImg, gt, imgName)
    % Compute difference image (XOR shows mismatched pixels)
    diffImg = xor(binaryImg, gt);
    diffSum = sum(sum(diffImg));
    
    % Check inverted polarity
    diffImgInv = xor(~binaryImg, gt);
    diffSumInv = sum(sum(diffImgInv));
    
    % Use correct polarity (whichever has fewer errors)
    if diffSumInv < diffSum
        fprintf('Note: Using inverted segmentation for %s\n', imgName);
        binaryImg = ~binaryImg;
        diffImg = diffImgInv;
        diffSum = diffSumInv;
    end
    
    accuracy = 100 * (1 - diffSum / numel(gt));
end

function displayOtsuResults(imgName, otsuLevel, diffSum, accuracy)
    fprintf('Image: %s\n', imgName);
    fprintf('Otsu Threshold: %.4f\n', otsuLevel);
    fprintf('Sum of Differences: %d pixels\n', diffSum);
    fprintf('Accuracy: %.2f%%\n\n', accuracy);
end


function visualizeOtsuThreshold(img, otsuLevel)
    % Histogram
    [counts, binLocations] = imhist(img);
    thresholdValue = otsuLevel * 255;

    % Figure
    figure();
    bar(binLocations, counts, 'FaceColor', [0.4 0.6 0.9], 'EdgeColor', 'none');
    hold on;
    yLimits = ylim;
    plot([thresholdValue thresholdValue], yLimits, 'r-', 'LineWidth', 2.5);
    
    % Labels
    xlabel('Pixel Intensity (0-255)', 'FontSize', 11);
    ylabel('Number of Pixels', 'FontSize', 11);
    title(sprintf('Pixel Distribution - Otsu Threshold = %.2f', thresholdValue),'FontSize', 12, 'FontWeight', 'bold');
    legend('Pixel Distribution', sprintf('Otsu Threshold (%.2f)', thresholdValue), 'Location', 'northeast');
    grid on;
    hold off;
end

function visualizeOtsuSegmentation(img, binaryImg, gt, diffImg, diffSum, otsuLevel)
    figure();
    subplot(2,2,1); imshow(img); title('Original');
    subplot(2,2,2); imshow(binaryImg); title(sprintf('Otsu (t=%.3f)', otsuLevel));
    subplot(2,2,3); imshow(gt); title('Ground Truth');
    subplot(2,2,4); imshow(diffImg); title(sprintf('Difference (%d pixels)', diffSum));
end

%% 2.1.b - Niblack's Local Thresholding

% Niblack parameters to test
kValues = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5];  
windowSizes = [11, 51, 101, 301, 501];  

% Entry point for processing image with a for loop
for i = 1:length(imageFiles)
    processImageWithNiblack(imageFiles{i}, gtFiles{i}, kValues, windowSizes);
end

function processImageWithNiblack(imgFile, gtFile, kValues, windowSizes)
    [img, gt] = loadAndPreprocessImages(imgFile, gtFile);
    [bestK, bestWindowSize, bestDiffSum, bestBinaryImg, allResults] = findBestNiblackParams(img, gt, kValues, windowSizes);
    displayNiblackResults(imgFile, bestK, bestWindowSize, bestDiffSum, numel(gt));
    plotParameterAnalysis(allResults, imgFile);
    visualizeNiblackSegmentation(img, bestBinaryImg, gt, bestDiffSum, bestK, bestWindowSize);
    visualizeNiblackThreshold3D(img, bestK, bestWindowSize)
end

function [bestK, bestWindowSize, minDiffSum, bestBinaryImg, allResults] = findBestNiblackParams(img, gt, kValues, windowSizes) 
    minDiffSum = inf;
    bestK = kValues(1);
    bestWindowSize = windowSizes(1);
    bestBinaryImg = [];
    
    % Store all results for analysis
    resultIdx = 1;
    allResults = struct('k', {}, 'windowSize', {}, 'diffSum', {});
    
    for k = kValues
        for winSize = windowSizes
            % Apply Niblack thresholding
            binaryImg = applyNiblackThresholding(img, k, winSize);
            [diffSum, binaryImg] = compareWithGroundTruthNiblack(binaryImg, gt);
            
            % Store
            allResults(resultIdx).k = k;
            allResults(resultIdx).windowSize = winSize;
            allResults(resultIdx).diffSum = diffSum;
            resultIdx = resultIdx + 1;
            
            % Update
            if diffSum < minDiffSum
                minDiffSum = diffSum;
                bestK = k;
                bestWindowSize = winSize;
                bestBinaryImg = binaryImg;
            end
        end
    end
end

function binaryImg = applyNiblackThresholding(img, k, windowSize)
    img = double(img);
    
    % Compute local mean using sliding window
    localMean = imboxfilt(img, windowSize);
    
    % Compute local standard deviation
    localMean2 = imboxfilt(img.^2, windowSize);
    localVar = localMean2 - localMean.^2;
    localStd = sqrt(max(localVar, 0));
    
    % Compute threshold
    threshold = localMean + k * localStd;
    binaryImg = img > threshold;
end

function [diffSum, binaryImg] = compareWithGroundTruthNiblack(binaryImg, gt)
    % Compute difference
    diffSum = sum(sum(xor(binaryImg, gt)));
    
    % Check inverted polarity
    diffSumInv = sum(sum(xor(~binaryImg, gt)));
    
    % Use correct polarity
    if diffSumInv < diffSum
        binaryImg = ~binaryImg;
        diffSum = diffSumInv;
    end
end

function displayNiblackResults(imgName, bestK, bestWindowSize, bestDiffSum, totalPixels)
    fprintf('\nImage: %s\n', imgName);
    fprintf('Best k: %.2f\n', bestK);
    fprintf('Best Window Size: %d\n', bestWindowSize);
    fprintf('Sum of Differences: %d pixels (%.2f%%)\n', ...
        bestDiffSum, 100 * bestDiffSum / totalPixels);
    fprintf('Accuracy: %.2f%%\n', 100 * (1 - bestDiffSum / totalPixels));
end

function plotParameterAnalysis(allResults, imgName)
    % Extract data
    kValues = [allResults.k];
    windowSizes = [allResults.windowSize];
    diffSums = [allResults.diffSum];
    
    % Get unique values
    uniqueK = unique(kValues);
    uniqueWinSizes = unique(windowSizes);
    
    % Create grid for heatmap
    [K, ~] = meshgrid(uniqueK, uniqueWinSizes);
    diffGrid = zeros(size(K));
    
    for i = 1:length(allResults)
        kIdx = find(uniqueK == allResults(i).k);
        winIdx = find(uniqueWinSizes == allResults(i).windowSize);
        diffGrid(winIdx, kIdx) = allResults(i).diffSum;
    end
    
    % Plot heatmap
    figure();
    imagesc(uniqueK, uniqueWinSizes, diffGrid);
    colorbar;
    xlabel('k value');
    ylabel('Window Size');
    title(sprintf('Difference Sum vs Parameters - %s', imgName));
    set(gca, 'YDir', 'normal');
    
    % Mark best point
    [~, minIdx] = min(diffSums);
    hold on;
    plot(kValues(minIdx), windowSizes(minIdx), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
    legend('Best Parameters');
end

function visualizeNiblackSegmentation(img, binaryImg, gt, diffSum, k, windowSize)
    diffImg = xor(binaryImg, gt);
    figure();
    subplot(2,2,1); imshow(img); title('Original');
    subplot(2,2,2); imshow(binaryImg); 
    title(sprintf('Niblack (k=%.2f, win=%d)', k, windowSize));
    subplot(2,2,3); imshow(gt); title('Ground Truth');
    subplot(2,2,4); imshow(diffImg); 
    title(sprintf('Difference (%d pixels)', diffSum));
end

function visualizeNiblackThreshold3D(img, k, windowSize)
    % Convert to double
    img = double(img);
    
    % Compute local statistics
    localMean = imboxfilt(img, windowSize);
    localMean2 = imboxfilt(img.^2, windowSize);
    localVar = localMean2 - localMean.^2;
    localStd = sqrt(max(localVar, 0));
    
    % Compute threshold map
    threshold = localMean + k * localStd;
    
    % Downsample for visualization
    [rows, cols] = size(threshold);
    step = 10;
    [X, Y] = meshgrid(1:step:cols, 1:step:rows);
    Z = threshold(1:step:rows, 1:step:cols);
    
    % Create 3D visualization
    figure();
    surf(X, Y, Z, 'EdgeColor', 'none');
    colormap('jet');
    colorbar;
    xlabel('X Position (pixels)', 'FontSize', 10);
    ylabel('Y Position (pixels)', 'FontSize', 10);
    zlabel('Threshold Value', 'FontSize', 10);
    title(sprintf('3D Threshold Surface (k=%.2f, win=%d)', k, windowSize),'FontSize', 12, 'FontWeight', 'bold');
    view(45, 30);
end
%% 2.1.c - Improvements to Niblack's Algorithm (Sauvola Method)

% Sauvola parameters to test
kValues_sauvola = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5];   
RValues_sauvola = [64, 96, 128, 160, 192];
windowSizes_sauvola = [11, 51, 101, 301, 501];  

% Entry point for processing image with a for loop
for i = 1:length(imageFiles)
    processImageWithSauvola(imageFiles{i}, gtFiles{i}, kValues_sauvola, RValues_sauvola, windowSizes_sauvola);
end

function processImageWithSauvola(imgFile, gtFile, kValues, RValues, windowSizes)

    [img, gt] = loadAndPreprocessImages(imgFile, gtFile);
    [bestK, bestR, bestWindowSize, bestDiffSum, bestBinaryImg, allResults] = findBestSauvolaParams(img, gt, kValues, RValues, windowSizes);
    displaySauvolaResults(imgFile, bestK, bestR, bestWindowSize, bestDiffSum, numel(gt));
    plotSauvolaParameterAnalysis3D(allResults, imgFile);
    visualizeSauvolaSegmentation(img, bestBinaryImg, gt, bestDiffSum, bestK, bestR, bestWindowSize);
    visualizeSauvolaThreshold3D(img, bestK, bestR, bestWindowSize);
end

function [bestK, bestR, bestWindowSize, minDiffSum, bestBinaryImg, allResults] = findBestSauvolaParams(img, gt, kValues, RValues, windowSizes)
    
    minDiffSum = inf;
    bestK = kValues(1);
    bestR = RValues(1);
    bestWindowSize = windowSizes(1);
    bestBinaryImg = [];
    
    % Store all results for analysis (3 parameters)
    resultIdx = 1;
    allResults = struct('k', {}, 'R', {}, 'windowSize', {}, 'diffSum', {});
    
    for k = kValues
        for R = RValues
            for winSize = windowSizes
                % Apply Sauvola thresholding
                binaryImg = applySauvolaThresholding(img, k, R, winSize);
                [diffSum, binaryImg] = compareWithGroundTruthSauvola(binaryImg, gt);
                
                % Store results
                allResults(resultIdx).k = k;
                allResults(resultIdx).R = R;
                allResults(resultIdx).windowSize = winSize;
                allResults(resultIdx).diffSum = diffSum;
                resultIdx = resultIdx + 1;
                
                % Update
                if diffSum < minDiffSum
                    minDiffSum = diffSum;
                    bestK = k;
                    bestR = R;
                    bestWindowSize = winSize;
                    bestBinaryImg = binaryImg;
                end
            end
        end
    end
end


function binaryImg = applySauvolaThresholding(img, k, R, windowSize)
    img = double(img);
    
    % Compute local mean 
    localMean = imboxfilt(img, windowSize);
    
    % Compute local standard deviation 
    localMean2 = imboxfilt(img.^2, windowSize);
    localVar = localMean2 - localMean.^2;
    localStd = sqrt(max(localVar, 0));
    
    % Compute threshold
    threshold = localMean .* (1 + k * ((localStd / R) - 1));
    binaryImg = img > threshold;
end

function [diffSum, binaryImg] = compareWithGroundTruthSauvola(binaryImg, gt)
    % Compute difference
    diffSum = sum(sum(xor(binaryImg, gt)));
    
    % Check inverted polarity
    diffSumInv = sum(sum(xor(~binaryImg, gt)));
    
    % Use correct polarity
    if diffSumInv < diffSum
        binaryImg = ~binaryImg;
        diffSum = diffSumInv;
    end
end

function displaySauvolaResults(imgName, bestK, bestR, bestWindowSize, bestDiffSum, totalPixels)
    fprintf('\nImage: %s\n', imgName);
    fprintf('Best k: %.2f\n', bestK);
    fprintf('Best R: %d\n', bestR);
    fprintf('Best Window Size: %d\n', bestWindowSize);
    fprintf('Sum of Differences: %d pixels (%.2f%%)\n', bestDiffSum, 100 * bestDiffSum / totalPixels);
    fprintf('Accuracy: %.2f%%\n', 100 * (1 - bestDiffSum / totalPixels));
end

function plotSauvolaParameterAnalysis3D(allResults, imgName)
    % Extract data
    kValues = [allResults.k];
    RValues = [allResults.R];
    windowSizes = [allResults.windowSize];
    diffSums = [allResults.diffSum];
    
    % Find best parameters
    [minDiff, minIdx] = min(diffSums);
    bestK = kValues(minIdx);
    bestR = RValues(minIdx);
    bestWin = windowSizes(minIdx);
    
    % Create 3D scatter plot
    figure('Position', [100, 100, 1000, 800]);
    
    % Color code by error (blue=good, red=bad)
    scatter3(kValues, RValues, windowSizes, 100, diffSums, 'filled', 'MarkerEdgeColor', 'k');
    colormap(jet);
    c = colorbar;
    c.Label.String = 'Error Count';
    
    xlabel('k value', 'FontSize', 12);
    ylabel('R value', 'FontSize', 12);
    zlabel('Window Size', 'FontSize', 12);
    title(sprintf('3D Sauvola Parameter Space - %s\nBest: k=%.1f, R=%d, win=%d', imgName, bestK, bestR, bestWin), 'FontSize', 14);
    
    % Mark best point
    hold on;
    scatter3(bestK, bestR, bestWin, 500, 'r', 'p', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 3);
    grid on;
    view(45, 30);
    
    % Add legend
    legend('Parameter combinations', 'Best parameters', 'Location', 'best');
end

function visualizeSauvolaSegmentation(img, binaryImg, gt, diffSum, k, R, windowSize)
    diffImg = xor(binaryImg, gt);
    figure();
    subplot(2,2,1); imshow(img); title('Original');
    subplot(2,2,2); imshow(binaryImg); 
    title(sprintf('Sauvola (k=%.2f, R=%d, win=%d)', k, R, windowSize));
    subplot(2,2,3); imshow(gt); title('Ground Truth');
    subplot(2,2,4); imshow(diffImg); 
    title(sprintf('Difference (%d pixels)', diffSum));
end

function visualizeSauvolaThreshold3D(img, k, R, windowSize)
    % Convert to double
    img = double(img);
    
    % Compute local statistics
    localMean = imboxfilt(img, windowSize);
    localMean2 = imboxfilt(img.^2, windowSize);
    localVar = localMean2 - localMean.^2;
    localStd = sqrt(max(localVar, 0));
    
    % Compute threshold map
    threshold = localMean .* (1 + k * ((localStd / R) - 1));
    
    % Downsample for visualization
    [rows, cols] = size(threshold);
    step = 10;
    [X, Y] = meshgrid(1:step:cols, 1:step:rows);
    Z = threshold(1:step:rows, 1:step:cols);
    
    % Create 3D visualization
    figure();
    surf(X, Y, Z, 'EdgeColor', 'none');
    colormap('jet');
    colorbar;
    xlabel('X Position (pixels)', 'FontSize', 10);
    ylabel('Y Position (pixels)', 'FontSize', 10);
    zlabel('Threshold Value', 'FontSize', 10);
    title(sprintf('3D Sauvola Threshold Surface (k=%.2f, R=%d, win=%d)', k, R, windowSize), 'FontSize', 12, 'FontWeight', 'bold');
    view(45, 30);
end

%% 3.1.a - Implementing disparity map algorithm

function disparityMap = computeDisparityMap(leftImg, rightImg, templateRows, templateCols)
    leftImg = double(leftImg);
    rightImg = double(rightImg);
    [rows, cols] = size(leftImg);
    disparityMap = zeros(rows, cols);
    
    maxDisparity = 15;
    halfTemplateRows = floor(templateRows / 2);
    halfTemplateCols = floor(templateCols / 2);
    
    % Precompute squared images for efficiency
    leftImg2 = leftImg.^2;
    rightImg2 = rightImg.^2;
    
    for y = (halfTemplateRows + 1):(rows - halfTemplateRows)
        for x = (halfTemplateCols + 1):(cols - halfTemplateCols)
            % Extract template from left image
            template = leftImg(y - halfTemplateRows : y + halfTemplateRows, x - halfTemplateCols : x + halfTemplateCols);
            minX = max(halfTemplateCols + 1, x - maxDisparity);
            
            % Precompute sum(G^2)
            sumG2 = sum(sum(leftImg2(y - halfTemplateRows : y + halfTemplateRows, x - halfTemplateCols : x + halfTemplateCols)));
            
            % Compute SSD
            bestSSD = inf;
            bestDisparity = 0;

            for xr = minX:x
                % Extract candidate from right image
                candidate = rightImg(y - halfTemplateRows : y + halfTemplateRows, xr - halfTemplateCols : xr + halfTemplateCols);
                sumI2 = sum(sum(rightImg2(y - halfTemplateRows : y + halfTemplateRows, xr - halfTemplateCols : xr + halfTemplateCols)));
                sumIG = sum(sum(template .* candidate));
                ssd = sumI2 + sumG2 - 2*sumIG;
                if ssd < bestSSD
                    bestSSD = ssd;
                    bestDisparity = x - xr;
                end
            end
            disparityMap(y, x) = -bestDisparity;
        end
    end
end


%% 3.1.b - Preprocessing pair images

left_corridor = imread('corridorl.jpg');
right_corridor = imread('corridorr.jpg');

function processed = preprocess_img(preprocess)
    if size(preprocess, 3) == 3
        processed = rgb2gray(preprocess);
    else
        processed = preprocess;
    end
end

processed_left_corridor = preprocess_img(left_corridor);
processed_right_corridor = preprocess_img(right_corridor);
figure();
imshow(processed_left_corridor);
figure();
imshow(processed_right_corridor);

%% 3.1.c - Obtaining the disparity map

D = computeDisparityMap(processed_left_corridor, processed_right_corridor, 11, 11);

figure();
imshow(-D, [-15 15]);
title('Disparity Map (corridor)');
colormap(gray);
colorbar;

gt_corridor = imread('corridor_disp.jpg');

figure();
imshow(gt_corridor);
title('Disparity Map (ground truth)');
colormap(gray);
colorbar;

%% 3.1.d - Rerun algorithm

left_triclopsi2 = imread('triclopsi2l.jpg');
right_triclopsi2 = imread('triclopsi2r.jpg');

processed_left_triclopsi2 = preprocess_img(left_triclopsi2);
processed_right_triclopsi2 = preprocess_img(right_triclopsi2);
figure();
imshow(processed_left_triclopsi2);
figure();
imshow(processed_right_triclopsi2);

D2 = computeDisparityMap(processed_left_triclopsi2, processed_right_triclopsi2, 11, 11);

figure();
imshow(-D2, [-15 15]);
title('Disparity Map (triclopsi)');
colormap("gray");
colorbar;

gt_triclopsi2 = imread('triclopsid.jpg');

figure();
imshow(gt_triclopsi2);
title('Disparity Map (ground truth)');
colormap(gray);
colorbar;








