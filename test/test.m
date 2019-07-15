%% 
clc;clear;close all;
%%
load('../data/test_x.mat');
load('../data/test_y.mat');
test_x=x_test;
test_y=y_test;
%%
data=squeeze(test_x(1,:,:));
%%
tic;
[status, cmdout] = system('python predict.py');
toc;