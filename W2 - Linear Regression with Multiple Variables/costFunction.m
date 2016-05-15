function [ J ] = costFunction( X, y, theta )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
m = size(X, 2);
predictions = theta'*X;
sqrErrors = (predictions - y) .^2;

J = 1/(2*m) * sum(sqrErrors);

end

