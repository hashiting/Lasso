function wstar = Lasso(Y,Phi,lambda)    
% input:
% Y    - outputs (column vector of size n)
% Phi  - design matrix (size n times D)
% lambda - regularization parameter (>=0)
%
% output:
% w               - optimal weight vector (size D, column vector) of the Lasso

[num,dim]=size(Phi);


% in the transformed optimization problem you have two variables wplus and
% wneg - both are restricted to be positive

wplus=rand(dim,1); % the old iterate x_t
wneg =rand(dim,1); 

wplusnew=rand(dim,1); % this is the new iterate (x_{t+1})
wnegnew=rand(dim,1);

counter=1;
% stopping criterion is here the norm of difference of two iterates
while( sqrt(norm(wplus-wplusnew)^2 + norm(wneg-wnegnew)^2)>1E-7)
  
  wplus = wplusnew;
  wneg  = wnegnew;
  
  % compute the gradient 
  [gradplus,gradneg]=GradLassoObjective(Y,Phi,lambda,wplus,wneg);
  
  % get stepsize
  stepsize=getStepSize(Y,Phi,lambda,wplus,wneg,gradplus,gradneg);
  
  % projected gradient steps
  wplusnew = ProjectionPositiveOrthant(wplus - stepsize*gradplus);
  wnegnew  = ProjectionPositiveOrthant(wneg - stepsize*gradneg);
  
  if(rem(counter,10)==0)
    Obj = LassoObjective(Y,Phi,lambda,wplusnew,wnegnew);  
    disp(['Iteration: ',num2str(counter),' - Current Objective: ',num2str(Obj,'%1.12f'),' - stepsize: ',num2str(stepsize)]);
  end
  counter=counter+1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% return the weight vector for the original Lasso problem
wstar  = [wplusnew wnegnew];





function Projw = ProjectionPositiveOrthant(w) % returns the projection of w onto the positive orthant
  Projw = max(w,0);

function [gradplus,gradneg] = GradLassoObjective(Y,Phi,lambda,wplus,wneg) % returns the gradient of the objective at (wplus,wneg)
  % gradplus is the gradient wrt wplus, gradneg wrt to wneg
  [num,dim]=size(Phi);
  gradplus = (2*Phi'*(Phi*wplus-Y-Phi*wneg))./num+ lambda;
  gradneg =  (2*Phi'*(Phi*wneg-Phi*wplus+Y))./num+ lambda;

function fval = LassoObjective(Y,Phi,lambda,wplus,wneg) % returns the objective of the optimization problem given wplus and wneg
  [num,dim]=size(Phi);
  fval = 1/num*norm(Y-Phi*(wplus-wneg))^2 + lambda*sum(wplus) + lambda*sum(wneg);
  
function stepsize=getStepSize(Y,Phi,lambda,wplus,wneg,gradplus,gradneg)  % given the current points and their gradients returns the stepsize
  stepsize=0.5; beta = 0.5;
  objective = LassoObjective(Y,Phi,lambda,wplus,wneg);
  wplusnew = ProjectionPositiveOrthant(wplus - stepsize*gradplus);
  wnegnew  = ProjectionPositiveOrthant(wneg - stepsize*gradneg);
  newobjective = LassoObjective(Y,Phi,lambda,wplusnew,wnegnew);
  
  % stepsize selection via backtracking line search 
  % (specific for projected gradient descent)
  while( newobjective > objective + gradplus'*(wplusnew-wplus)+gradneg'*(wnegnew-wneg)+1/(2*stepsize)*(norm(wplusnew-wplus)^2+norm(wnegnew-wneg)^2))
   stepsize=beta*stepsize;
   wplusnew = ProjectionPositiveOrthant(wplus - stepsize*gradplus);
   wnegnew  = ProjectionPositiveOrthant(wneg - stepsize*gradneg);
   newobjective = LassoObjective(Y,Phi,lambda,wplusnew,wnegnew);
  end 
 
  