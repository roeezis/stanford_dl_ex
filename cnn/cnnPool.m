function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
    convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region.
%
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
mean_filter = ones(poolDim)/(poolDim^2);
featureIdx = 1:poolDim:(convolvedDim - max(0, poolDim - 1));
% fun = @(block_struct) mean2(block_struct.data) * ones(size(block_struct.data));
% subDim = convolvedDim / poolDim;
for imageNum = 1:numImages
    for filterNum = 1:numFilters
        features = squeeze(convolvedFeatures(:, :, filterNum, imageNum));
        mean_features = conv2(features,mean_filter,'valid');
        %         mean_features = blockproc(features,[poolDim poolDim],fun);
        pooledFeatures(:, :, filterNum, imageNum) = ...
            mean_features(featureIdx,featureIdx);
        %           reshape(unique(mean_features), subDim,[]);
    end
end

end

