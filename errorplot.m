fileID = fopen('mnist_5_14_32.txt','r');
intro = textscan(fileID,'%s',:,'Delimiter','\n');
disp(intro)