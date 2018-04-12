clear;
%Multiclass boosting using SAMME algorithm
% [trainingImages, trainingLabels, testImages, testLabels] = helperCIFAR10Data.load('.');
% [w,h,~,N] = size(trainingImages);
% trainingImages = double(reshape(permute(trainingImages(:,:,1,:),[4 2 3 1]),[],w*h)./255);
% testImages = double(reshape(permute(testImages(:,:,2,:),[4 2 3 1]),[],w*h)./255);
% trainingLabels = double(trainingLabels+1); testLabels=double(testLabels+1);

%[trainImages, trainLabels, testImages, testLabels] = newCIFAR10Data.load('.');
trainImages = loadMNISTImages('mnist/train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');
testImages = loadMNISTImages('mnist/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
trainImages = double(trainImages');
testImages = double(testImages');
trainLabels = double(trainLabels+1); testLabels=double(testLabels+1);
[N,M] = size(trainImages);

K=10;
%weight = sparse(1:N,1:N,ones(N,1))/N;K=10;
weight = ones(N,1)/N;
%specify result file and output format
resultFile = fopen('result.txt','w');

for i=1:500
    [fg, missed] = PCABayes(trainImages, trainLabels, weight,17,16);
    fprintf(resultFile,'error=%9.7f and alpha=%9.7f\n',fg.error,fg.alpha);
    fprintf('error=%9.7f and alpha=%9.7f\n',fg.error,fg.alpha);
    models{i} = fg; %#ok<SAGROW>
    for j = 1:N
        if missed(j) == 1
            weight(j) = weight(j)*(K-1)/(K * fg.error);
        else
            weight(j) = weight(j)/(K * (1 - fg.error));
        end
    end
    acc = benchMark(testImages, testLabels, models);
    fprintf(resultFile,'Boosting rounds %4d, Test accuracy: %9.7f\n',size(models,2),acc);
    fprintf('Boosting rounds %3d, Test accuracy: %9.7f\n',size(models,2),acc);
end

function arate = benchMark(timgs, lbs, models)
    num = length(timgs);
    acc = zeros(1,num);
    parfor i = 1:num
        score = zeros(1,10);
        for j = 1:size(models,2)
            pb = models{j};
            pred = pb.predict(timgs(i,:));
            score(pred) = score(pred) + pb.alpha;
        end
        [~,lb] = max(score);
        acc(i) = lbs(i) == lb;
    end
    arate = sum(acc) / num;
end
