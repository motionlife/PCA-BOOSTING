classdef PCABayes2
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    properties
        K
        yDist
        factors
        xCondDist
        error
        alpha
        eigVectors  
    end
    
    methods
        function [obj,missed]= PCABayes2(images,labels,weight,degree,leaf)
            %UNTITLED Construct an instance of this class
            %Detailed explanation goes here
            obj.K = 10;
            obj.yDist = getLabelDistr(obj,labels,weight);
            obj.factors = selectFactors(obj,1:length(images(1,:)),degree,leaf);
            obj.eigVectors = getWeightedEigenVectors(obj,images,weight,degree,leaf);
            obj.xCondDist = getLeafCondDistr(obj,images,labels,weight);%?weighted once more???
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
            DIM = 5;
            vectors = zeros(leaf,DIM,degree);
            for i = 1:degree
                [vectors(:,:,i),~] = eigs(weightedcov(images(:,obj.factors(i,:)),weight),DIM);
            end
        end
        
        function distr = getLeafCondDistr(obj,images,labels,weight)
            [~, DIM, degree] = size(obj.eigVectors);
            N = size(images,1);
            projections = zeros(N,degree,DIM);
            for i=1:N
                projections(i,:,:) = project(obj,images(i,:),degree,DIM);
            end
            distr = cell(1,obj.K);%TODO: IS IT NECCESSARY TO CALCULATE WEIGHTED DISTRUBUTION?
            for i = 1:obj.K
                distr{i}.mu = zeros(degree,DIM);
                distr{i}.sigma = zeros(DIM,DIM,degree);
                for j = 1:degree
                    mt = squeeze(projections(labels==i,j,:));
                    wt = weight(labels==i);
                    distr{i}.mu(j,:) = sum(mt.*(wt(:) / sum(wt)));
                    distr{i}.sigma(:,:,j) = weightedcov(mt,wt);
                end
            end
        end
        
        function ft = project(obj,img,degree,DIM)
            ft = zeros(degree,DIM);
            for i=1:degree
                ft(i,:) = img(obj.factors(i,:)) * obj.eigVectors(:,:,i);
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
            [~, DIM, degree] = size(obj.eigVectors);
            x = project(obj,img,degree,DIM);
            score = zeros(1,obj.K);
            for i = 1:obj.K
                pdf = [mvnpdf(x,obj.xCondDist{i}.mu,obj.xCondDist{i}.sigma); obj.yDist(i)];
                score(i) = sum(log(pdf));
            end
            [~,result] = max(score);
        end
    end
end

