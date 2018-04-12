classdef PCABayes
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    properties
        K
        labelDist
        factors
        leafCondDistr
        error
        alpha
        eigVectors  
    end
    
    methods
        function [obj,missed]= PCABayes(images,labels,weight,degree,leaf)
            %UNTITLED Construct an instance of this class
            %Detailed explanation goes here
            obj.K = 10;
            obj.labelDist = getLabelDistr(obj,labels,weight);
            obj.factors = selectFactors(obj,1:length(images(1,:)),degree,leaf);
            obj.eigVectors = getWeightedEigenVectors(obj,images,weight,degree,leaf);
            obj.leafCondDistr = getLeafCondDistr(obj,images,labels);%?weighted once more???
            [obj.error,missed] = getError(obj,images,labels,weight);
            obj.alpha = log((1/obj.error - 1)*(obj.K-1));
        end
        
        function margin = getLabelDistr(obj,labels,weight)
            margin = zeros(1,obj.K);
            for i = 1: obj.K
                margin(i) = sum(weight(labels==i));
            end
        end
        
        function factors =  selectFactors(~,nodes,degree,leaf)
            factors = zeros(degree,leaf);
            for i=1:degree
                p = datasample(nodes,leaf,'Replace',false);
                nodes = setdiff(nodes,p);
                factors(i,:) = p;
            end
        end
        
        function vectors = getWeightedEigenVectors(obj,images,weight,degree,leaf)
            vectors = zeros(degree,leaf);
            for i = 1:degree
                [vectors(i,:),~] = eigs(weightedcov(images(:,obj.factors(i,:)),weight),1);
            end
        end
        
        function distr = getLeafCondDistr(obj,images,labels)
            degree = size(obj.factors,1);
            N = size(images,1);
            projections = zeros(N,degree);
            for i=1:N
                projections(i,:) = project(obj,images(i,:));
            end
            distr = zeros(degree,2,obj.K);%col1=weighted mean, col2=weighted variance
            for i=1:obj.K
                for j = 1:degree
                    distr(j,1,i) = mean(projections(labels==i,j));
                    distr(j,2,i) = var(projections(labels==i,j));
                end
            end
        end
        
        function ft = project(obj,x)
            degree = size(obj.factors,1);
            ft = zeros(1,degree);
            for i=1:degree
                ft(i) = dot(x(obj.factors(i,:)),obj.eigVectors(i,:));
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
            degree = size(obj.factors,1);
            x = project(obj,img);
            score = zeros(1,obj.K);
            for i = 1:obj.K
                prob = log(obj.labelDist(i));
                for j = 1:degree
                    mu = obj.leafCondDistr(j,1,i);
                    sigma = sqrt(obj.leafCondDistr(j,2,i));
                    prob = prob + log(normcdf(x(j)+sigma/2,mu,sigma) - normcdf(x(j)-sigma/2,mu,sigma));
                end
                score(i) = prob;
            end
            [~,result] = max(score);
        end
        
    end
end

