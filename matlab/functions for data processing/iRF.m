function F=iRF(X,Y,N,w,Q,A,method)

% X: The data matrix of size m x p
% Y: The reponse vector of size m x 1
% N: the number of iterations
% w: the fixed window size to move over the whole spectra
% Q: the initialized number of sub-intervasl.
% A: the maximal principle component.
% method: pretreatment method.

tic;
[Mx,Nx]=size(X);
k=1;
% Obtain all the possible intervals 
for j=1:Nx-w+1
    intervals{1,k}=j:j+w-1;
    k=k+1;     
end   
 p=length(intervals);
 varIndex=1:p; 
 Q0=Q;
 % Initialize a subset of intervals: V0.     
 perm=randperm(p);     
 V0=perm(1:Q);   
 % Main loop for iRF   
 probability=zeros(1,p);
for i=1:N
  nVstar=min(p,max([round(randn(1)*0.3*Q+Q) 2]));
  if length(V0)==1
      U0=intervals{V0};
  end
  if length(V0)>=2
      U0=intervals{V0(1)};
      for iii=2:length(V0)
          % The union of the intervals
          U0=union(U0,intervals{V0(iii)});
      end
  end
  PLS=pls(X(:,U0),Y,A,method);    
  B=abs(PLS.coef_origin(1:end-1,end));
  clear Binterval;
  for ii=1:length(V0)
  [C,IA,IB] = intersect(U0,intervals{1,V0(ii)});
  % Sum of absoulte regression coefficient   
  Binterval(ii)=sum(B(IA));
  end
  nV1=nVstar;
  % Define a function
  Vstar=generateNewModel(V0,nV1,Binterval,varIndex,intervals,X,Y,A,method); 
  if length(V0)==1
      U0=intervals{V0};
  end
  if length(V0)>=2
      U0=intervals{V0(1)};
      for iii=2:length(V0)
          U0=union(U0,intervals{V0(iii)});
      end
  end
  CV0=plscvfold(X(:,U0),Y,A,10,method,0);
  ARMSEP0=CV0.RMSECV;
  if length(Vstar)>=2
      Ustar=intervals{Vstar(1)};
      for iii=2:length(Vstar)
          Ustar=union(Ustar,intervals{Vstar(iii)});
      end
  end
  CVstar=plscvfold(X(:,Ustar),Y,A,10,method,0);
  ARMSEPstar=CVstar.RMSECV;
  
  if ARMSEPstar<ARMSEP0
      probAccept=1;
  else
      probAccept=0.1*ARMSEP0/ARMSEPstar;
  end
  randJudge=rand(1);
  if probAccept>randJudge
    V0=Vstar;RMSEP(i)=ARMSEPstar;nIntervals(i)=nVstar;
  else
    V0=V0;RMSEP(i)=ARMSEP0;nIntervals(i)=Q;
  end
  probability(V0)=probability(V0)+1;
  Q=length(V0);
  if mod(i,100)==0;fprintf('The %dth sampling of iRF has finished.\n',i);end
end
probability=probability/N;
[sorted,Intervalsrank]=sort(-probability);
top10=Intervalsrank(1:10);
toc;
% Output
F.intervals=intervals;
F.N=N;
F.Q=Q0;
F.minutes=toc/60;
F.method=method;
F.Intervalsrank=Intervalsrank;  % All the ranked intervals.
F.top10=top10;
F.probability=probability;
F.nIntervals=nIntervals;
F.RMSEP=RMSEP;  
     
%% Plot
% Change in RMSEP when entering the first n intervals
figure;
subplot(2,1,1);
plot(1:F.N,F.nIntervals);
xlabel('Number of iterations');
ylabel('Number of intervals selected for a single iteration');
subplot(2,1,2);
plot(1:F.N, F.RMSEP);
xlabel('Number of iterations');
ylabel('RMSEP');
% Selection probability for each wavelength interval (circle the 10 intervals with the highest probability)
figure;
plot(1:length(F.intervals), F.probability);hold on;
plot(F.top10, F.probability(F.top10),'ro');
xlabel('band intervals');
ylabel('probability of selection');
title('Selection probability for each band interval');

%% Subfunctions
function Vstar=generateNewModel(V0,nV1,Binterval,varIndex,intervals,X,Y,A,method)
nV0=length(V0);
d=nV1-nV0;
if d>0
   
  varIndex(V0)=[];
  kvar=length(varIndex);
  perm=randperm(kvar);
  perm=perm(1:min(3*d,kvar));
  clear Vstartemp;
  Vstartemp=[V0 varIndex(perm)];
  if length(Vstartemp)==1
      Ustar=intervals{Vstartemp};
  end
   if length(Vstartemp)>=2
      Ustar=intervals{Vstartemp(1)};
      for iii=2:length(Vstartemp)
          Ustar=union(Ustar,intervals{Vstartemp(iii)});
      end
  end
  PLS=pls(X(:,Ustar),Y,A,method);
     B=abs(PLS.coef_origin(1:end-1,end));
  clear Binterval_star;
  for ii=1:length(Vstartemp)
  [C,IA,IB]=intersect(Ustar,intervals{1,Vstartemp(ii)});
  Binterval_star(ii)=sum(B(IA))/length(B(IA));
  end
  clear index;
  [sorted,index]=sort(-Binterval_star);
  Vstar=Vstartemp(index(1:nV1));
end
if d<0
   clear index;
  [sorted,index]=sort(-Binterval);  
  Vstar=V0(index(1:nV1));
end
 if d==0;
  Vstar=V0;
 end
