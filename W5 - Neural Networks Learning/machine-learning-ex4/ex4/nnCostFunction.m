function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = [ones(m, 1), X]; %5000x401
y_recoded = zeros(m, num_labels); % 5000x10
for j = 1:size(y,1)
    y_recoded(j, y(j)) = 1;
end

predictions2 = sigmoid(X*Theta1'); % 5000x25
predictions2 = [ones(m, 1), predictions2]; % 5000x26

predictions3 = sigmoid(predictions2*Theta2'); % 5000x10

errors = zeros(m, 1);
for t = 1:m
    errors(t) = y_recoded(t, :)*log(predictions3(t, :)') + (1 - y_recoded(t, :))*log(1-predictions3(t, :)');
end;

regularization = 0;
for s1 = 1:hidden_layer_size
    for s2 = 2:input_layer_size+1
        regularization = regularization + Theta1(s1,s2)^2;
    end;
end;

for s1 = 1:num_labels
    for s2 = 2:hidden_layer_size+1
        regularization = regularization + Theta2(s1,s2)^2;
    end;
end;

J = -1/m * sum(errors) + lambda/(2*m) * regularization;

% -------------------------------------------------------------
delta3 = predictions3 - y_recoded; % 5000x10

            % 5000x26                % 5000x26
delta2 = (Theta2' * delta3')' .* sigmoidGradient([ones(m, 1), X * Theta1']); % 5000x26 

Delta1 = zeros(hidden_layer_size,input_layer_size + 1);
              % 25x5000         % 5000x401
Delta1 = Delta1 + delta2(:, 2:end)' * X; %25x401

Delta2 = zeros(num_labels, hidden_layer_size + 1);
                % 10x5000      % 5000x26
Delta2 = Delta2 + delta3' * predictions2; %10x26

Theta1_grad = 1/m * Delta1;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m * Theta1(:, 2:end);

Theta2_grad = 1/m * Delta2;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m * Theta2(:, 2:end);
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
