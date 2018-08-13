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

%hypothesis = sigmoid(X*theta);
%J = (1/m)*sum(-y.*log(hypothesis)-(1-y).*log(1-hypothesis))+(lambda/(2*m))*((theta.*(theta+1).*(2.*theta+1))/6-theta(1,1)^2);
%grad(1,1) = 1/m*sum((hypothesis-y));
%for j= 2:length(theta)
%    grad(j,1) = 1/m*sum((hypothesis-y).*X(:,j))+(lambda/m)*theta(j,1);
 
hypothesis = sigmoid(X*theta);
sumOfTheta=0;
for i=2:length(theta)
    sumOfTheta=sumOfTheta+theta(i,1)^2;

J = (1/m)*sum(-y.*log(hypothesis)-(1-y).*log(1-hypothesis))+(lambda/(2*m)*sumOfTheta);
grad(1,1) = 1/m*sum((hypothesis-y));
for j= 2:length(theta)
    grad(j,1) = 1/m*sum((hypothesis-y).*X(:,j))+(lambda/m)*theta(j,1);
end





% =============================================================

end
