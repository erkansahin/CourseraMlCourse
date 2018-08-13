function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hypothesis = X*theta;

sumSquaredError  = (1/(2*m))*sum((hypothesis-y).^2);
regularizationError = 0;
for i=2:size(theta,1)
    regularizationError = regularizationError+ (lambda/(2*m))*theta(i)^2;
end
J = sumSquaredError+regularizationError;

grad(1)= sum(hypothesis-y)*(1/m);

% grad(2:end,:) = (1/m)*sum((hypothesis-y).*X(:,2))+(lambda/m)*theta(2:end);

for j= 2:length(theta)
    grad(j,1) = 1/m*sum((hypothesis-y).*X(:,j))+(lambda/m)*theta(j,1);
end








% =========================================================================

grad = grad(:);

end
