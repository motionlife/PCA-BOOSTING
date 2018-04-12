classdef RandomTree < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    properties
        K
        labelMargin
        factors
        distr
        error
        alpha
        eigens  
    end
    
    methods
        function [obj,missed]= RandomTree(images,labels,weight,degree)
            %UNTITLED Construct an instance of this class
            %Detailed explanation goes here
            obj.K = 10;
            [w,h,~,~] = size(images(:,:,:,1));
            obj.labelMargin = getLabelMargin(obj,labels,weight);
            obj.factors = selecFactors(obj,degree,w,h);
            obj.eigens = getWeightedEigenvectors(obj,images,weight);
            features = mapToFeature(obj,images);
            obj.distr = getDistr(obj,features,labels);
            [obj.error,missed] = getError(obj,features,labels,weight);
        end
        
        function margin = getLabelMargin(obj,labels,weight)
            margin = zeros(1,obj.K);
            for i = 1: obj.K
                margin(i) = trace(weight(labels==i,labels==i));
            end
        end
        
        function factors = selecFactors(~,degree,w,h)
            L = 3;
            factors = zeros(2,L,degree);
            for i=1:degree
                factors(:,:,i) = [randi(w-L+1)+(0:L-1);randi(h-L+1)+(0:L-1)];
            end
        end
        
        function eigens = getWeightedEigenvectors(obj,images,weight)
            CHANNEL = 2;% Only use the green channel data
%             images=images(:,:,:,1000);
            [~,l,degree] = size(obj.factors);
            eigens = zeros(l*l,degree);
            for i=1:degree
                pat = obj.factors(:,:,i);
                X = reshape(permute(images(pat(1,:),pat(2,:),CHANNEL,:),[4 2 3 1]),[],l*l);
                X = X - sum(X.*diag(weight));  %Is this the correct way to do weighted PCA?
                [eigens(:,i),~] = eigs(X'*weight*X,1);
%                 [eigens(:,i),~] = eigs(weightedcov(X,diag(weight)),1);
                %[U,Sig,~] = svd(X);
                %pca1(:,i) = U(:,1)*(-Sig(1));%map to the largest principle component
            end
        end
        
        function Pca = mapToFeature(obj,images)
            N = size(images,4);
            degree = size(obj.factors,3);
            Pca = zeros(N,degree);
            for i=1:N
                Pca(i,:) = extractFeature(obj,images(:,:,:,i));
            end
        end
        
        function f = extractFeature(obj,img)
            CHANNEL = 2;% Only use the green channel data
            degree = size(obj.factors,3);
            f = zeros(1,degree);
            for i=1:degree
                pat = obj.factors(:,:,i);
                vec = reshape(img(pat(1,:),pat(2,:),CHANNEL)',1,[]);
                f(i) = dot(vec,obj.eigens(:,i));%map to the projection on the eigenvector
            end
        end
        
        function distr = getDistr(obj,Pca,labels)
            degree = size(obj.factors,3);
            distr = zeros(2,degree,obj.K);%col1=weighted mean, col2=weighted variance
            for i=1:obj.K
                ft = Pca(labels==i,:);
                distr(1,:,i) = mean(ft);
                distr(2,:,i) = var(ft);
            end
        end
        
        function [err,missed] = getError(obj,Pca,labels,weight)
            len = size(Pca,1);
            err = 0;
            missed = zeros(1,len);
            for i = 1:len
                if predict(obj,Pca(i,:)) ~= labels(i)
                    err = err + weight(i,i);
                    missed(i) = 1;%cache the result for updating weight
                end
            end
            obj.alpha = log((1/err - 1)*(obj.K-1));
        end
        
        function result = predict(obj,X)
            degree = size(obj.factors,3);
            score = zeros(1,obj.K);
            for i = 1:obj.K
                prob = (1 - degree) * log(obj.labelMargin(i));
                for j = 1:degree
                    x = X(j);
                    mu = obj.distr(1,j,i);
                    sigma = sqrt(obj.distr(2,j,i));
                    prob = prob + log(normcdf(x+sigma/2,mu,sigma) - normcdf(x-sigma/2,mu,sigma));
                end
                score(i) = prob;
            end
            [~,result] = max(score);
        end
        
    end
end

