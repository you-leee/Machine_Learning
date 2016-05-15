function [ theta ] = gradientDescent( X, y, theta, alpha )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
m = size(X, 2);
predictions = theta'*X;
delta = sum((predictions - y) * X');

theta = theta - alpha * 1/m * delta;

end

