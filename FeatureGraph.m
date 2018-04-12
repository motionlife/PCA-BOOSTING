classdef FeatureGraph < handle
    %CLUSTER graph model used in each rount of boosting
    
    properties
        alpha
        Wml
    end
    
    methods (Static)
       function designMatrix = getDesignMatrix(images)
            %get the desgin matrix from data and weight: transform the pixel value to corresponding feature matrix
            [hight, width, ~, num] = size(images);
            designMatrix = zeros(num, 1 + hight*width + (hight-1)*width + hight*(width-1));
            for i = 1 : num
                designMatrix(i,:) = FeatureGraph.kernel(images(:,:,:,i));
            end
       end
       
       function feature = kernel(image)
            gs = double(rgb2gray(image));
            prod_col = gs(:,1:end-1).*gs(:,2:end);
            prod_row = gs(1:end-1,:).*gs(2:end,:);
            feature = [1 gs(:)' prod_col(:)' prod_row(:)'];
       end
    end
    
    methods
        function [obj,err,missed]= FeatureGraph(X, Y, W ,reg)
            %get parameters though closed form solution: (XT*W*X)\*XT*W*Y,
            %where W is a diagonal matrix containing weights of data
            %obj.Wml = (X' * W * X) \ ( X' * W * Y); % Without regularization
            temp = X' * W;
            obj.Wml =(reg + temp * X) \ (temp * Y); % With regularization
            [err, missed] = getErrorRate(obj,X,Y,W);
            obj.alpha = log((1/err - 1)*9);%obj.alpha = log(1/err - 1);%k=2;
        end
        
        function [err,missed] = getErrorRate(obj,X,Y,W)
            n = size(X,1);
            missed = zeros(1,n);
            err = 0;
            for i = 1:n
                [~, idx] = max(X(i,:) * obj.Wml);
                if find(Y(i,:)==1) ~= idx
                   missed(i) = 1;
                   err = err + W(i,i);
                end
            end
        end
    end
end

