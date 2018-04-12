clear;
%Multiclass boosting using SAMME algorithm
[trainingImages, trainingLabels, testImages, testLabels] = helperCIFAR10Data.load('.');

% selected = vertcat(find(testLabels == 3), find(testLabels == 4));
% trainingImages =  trainingImages(:,:,:,selected); trainingLabels = trainingLabels(selected);
% testImages =  testImages(:,:,:,selected); testLabels = testLabels(selected);
trainingLabels = double(trainingLabels); testLabels=double(testLabels);

Y = ones(length(trainingLabels),10);
for l=0:9
    Y(trainingLabels ~= l,l+1) = 0;
end

K = 10;
X = FeatureGraph.getDesignMatrix(trainingImages);
[m,n] = size(X);
W = sparse(1:m,1:m,ones(m,1))/m;
reg = sparse(1:n,1:n,ones(n,1)) * 1.777;

%specify result file and output format
resultFile = fopen('result.txt','w');

for i = 1:500
    [fg, error, missed] = FeatureGraph(X, Y, W, reg);
    fprintf(resultFile,'error=%9.7f and alpha=%9.7f\n',error,fg.alpha);
    fprintf('error=%9.7f and alpha=%9.7f\n',error,fg.alpha);
    models{i} = fg; %#ok<SAGROW>
    %disp(sum(diag(W)));
    for j = 1:m
        if missed(j) == 1
           W(j,j) = W(j,j)*(K-1)/(K * error);%W(j,j) = W(j,j)/(2*error);%
        else
           W(j,j) = W(j,j)/(K * (1 - error));%(j,j) = W(j,j)/(2*(1-error));%
        end
    end
    
    acc = benchMark(testImages, testLabels, models);
    fprintf(resultFile,'Boosting rounds %4d, Test accuracy: %9.7f\n',size(models,2),acc);
    fprintf('Boosting rounds %3d, Test accuracy: %9.7f\n',size(models,2),acc);
end


function acc = benchMark(Images, Labels, Models)
    num = size(Images,4);
    acc = 0;
    for i = 1:num
        feature = FeatureGraph.kernel(Images(:,:,:,i));
        score = zeros(1,10);
        for j=1:size(Models,2)
            [~,pred] = max(feature * Models{j}.Wml);
            score(pred) = score(pred) + Models{j}.alpha;
        end
        [~,idx] = max(score);
        acc = acc + (Labels(i) == idx-1);
    end
    acc = acc / num;
end

