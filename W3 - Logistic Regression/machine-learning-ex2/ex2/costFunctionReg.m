function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
predictions = sigmoid(X*theta);
errors = y'*log(predictions) + (1 - y)'*log(1-predictions);

tempTheta = theta(2:end);
regularization = lambda/(2*m) * sum(tempTheta.^2);

J = -1/m * sum(errors) + regularization;


temp = (predictions - y);
tempVal = theta;

tempVal(1,1) = 1/m * sum(temp.*X(:,1));
for i=2:length(theta)
    tempVal(i,1) = 1/m*sum(temp.*X(:,i)) + lambda/m * theta(i);
end

grad = tempVal;


% =============================================================

end
