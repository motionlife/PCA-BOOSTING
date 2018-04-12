classdef PCABayes4
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    properties
        K
        yDist
        xCondDist
        error
        alpha
        eigVectors
    end
    
    methods
        function [obj,missed]= PCABayes4(images,labels,weight,dim)
            obj.K = 10;
            obj.yDist = getLabelDistr(obj,labels,weight);
            obj.eigVectors = getEigVectors(obj,images,labels,weight,dim);
            obj.xCondDist = getCondDistr(obj,images,labels);%?weighted once more???
            [obj.error,missed] = getError(obj,images,labels,weight);
            obj.alpha = log((1/obj.error - 1)*(obj.K-1));
        end 
        function margin = getLabelDistr(obj,labels,weight)
            margin = zeros(1,obj.K);
            for i = 1: obj.K
                margin(i) = sum(weight(labels==i));
            end
        end
        function evs = getEigVectors(obj,images,labels,weight,dim)
            evs = zeros(size(images,2),dim,obj.K);
            for i = 1:obj.K
                %[evs(:,:,i),~] = eigs(weightedcov(images(labels==i,:),weight(labels==i)),dim);
                [V,D] = eig(weightedcov(images(labels==i,:),weight(labels==i)));
                pd = cumsum(diag(D)/sum(diag(D)));
                evs(:,:,i) = V(:, arrayfun(@(r)find(r<pd,1),rand(1,dim)));
            end
        end     
        function distr = getCondDistr(obj,images,labels)
            distr = cell(1,obj.K);%TODO: IS IT NECCESSARY TO CALCULATE WEIGHTED DISTRUBUTION?
            for i = 1:obj.K
                %mt = images(labels==i,:) * obj.eigVectors(:,:,i);
                mt = normr(images(labels==i,:) * obj.eigVectors(:,:,i));
                %wt = weight(labels==i);
                %distr{i}.mu = sum(mt.*(wt(:) / sum(wt)));
                distr{i}.mu = mean(mt);
                distr{i}.sigma = sqrt(var(mt)); %distr{i}.sigma = weightedcov(mt,wt);
            end
        end
        function [err,missed] = getError(obj,images,labels,weight)
            len = length(labels);
            missed = zeros(1,len);
            errs = zeros(1,len);
            parfor i = 1:len
                if predict(obj,images(i,:)) ~= labels(i)
                    errs(i) = weight(i);
                    missed(i) = 1;%cache the result for updating weight
                end
            end
            err = sum(errs);
        end
        function result = predict(obj,img)
            score = zeros(1,obj.K);
            for i = 1:obj.K
                x = normr(img * obj.eigVectors(:,:,i));
                %pdf = [mvnpdf(x,obj.xCondDist{i}.mu,obj.xCondDist{i}.sigma); obj.yDist(i)];
                pdf = [normpdf(x,obj.xCondDist{i}.mu,obj.xCondDist{i}.sigma) obj.yDist(i)];
                score(i) = sum(log(pdf));
            end
            [~,result] = max(score);
        end
    end
end