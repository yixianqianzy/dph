function [B, D, nn] =train_sae_pair(sp, content,opts, sizes, Train, Test, cold_item, row,cold_rating, r, alpha, beta, lammda, times);
% [M,t]=size(content);
[N,M]=size(Train);
sae = saesetup(sizes);
sae = saetrain(sae, content, opts);
nn = nnsetup([sizes, r]);
nn.W{1} = sae.ae{1}.W{1};
% nn.W{2} = sae.ae{2}.W{1};
nn = nnf1(nn, content);
%% Initialization
D=sign(nn.a{nn.n})';
rand('state',0);
B=rand(r,N);
B(B>0.5)=1; B(B<=0.5)=-1;
rand('state',0);
X=randn(r,N);
Y=randn(r,M);
%% Training
%Rt=(Train>0);


for u=1:N
    pos = sum(Train(u,:));
    if pos > 0
        Ia(u)=1./pos;  %I_u^+, the number of positive ratings of user u
        Is(u)=1./(M-pos);
    else
        Ia(u)=0;
        Is(u)=0;
    end
end
% Is=1./(M-sum(Train,2));  % the number of negative ratings of user u


% Loss = zeros(1, times);
%  L1= Loss;
%  L2= Loss;
%  L3= Loss;

   fid=fopen('C:\Users\13184888\Desktop\cae_tkde\dae_pair\sp_acc_result_times.txt','a');
   fprintf(fid,'sp=%f  ',sp);
    fprintf(fid,'\n');
       fprintf(fid,'lammda=%f  ',lammda);
    fprintf(fid,'\n');
    fclose(fid);
  fid=fopen('C:\Users\13184888\Desktop\cae_tkde\dae_pair\cd_acc_result_times.txt','a');
   fprintf(fid,'sp=%f  ',sp);
    fprintf(fid,'\n');
           fprintf(fid,'lammda=%f  ',lammda);
    fprintf(fid,'\n');
    fclose(fid);
for t=1:times
    B  = Bsub1( Train, B, D, X,r, N, M, Ia', Is', alpha);
    D =Dsub1(Train, B, D, Y, r, Ia', Is', N, M, beta, nn.a{nn.n}, lammda);
    nn.learningRate = 0.0002;
    opts.numepochs =1;
    [nn,L] = nntrain(nn, content, D', opts);
%     [nn,L]=fint_tune(nn,opts,M,content,D');
    nn = nnf1(nn, content);
    X = UpdateSVD(B);
    Y = UpdateSVD(D);
  %  plot(L);
%    gre = find(Train);
%   L00= zeros(N,M);
%   L01= 2*Train*r - r - B'*D;
%    L00(gre) = L01(gre);
%    L1(t)  = norm(L00, 'fro')^2;
 
%%calculate the loss function: correct
% L10 = zeros(N,1);
% num_u  = 0;
%  for u=1:N
%       pos_u=find(Train(u,:));
%       neg_u = setdiff(1:M,pos_u);
%       Iu = length(pos_u);
%       if Iu>0
%       Ias = M-Iu;
%       Da = D(:, pos_u);
%        Ds = D(:, neg_u );
%          loo = B(:,u)'*Da;
%           ls0= 2*r - loo;
%            ls1= norm(ls0)^2;
%            lj0 =   B(:,u)'*Ds;
%            lj1 = Iu*norm(lj0)^2;
%           lij0 =  sum(B(:,u)'*Ds);
%            lij1 =Iu* 4*r*lij0;
%            luij1 = 2*sum(loo)*lij0;
%            L10(u) = (ls1+lj1+lij1-luij1) / (Iu*Ias);
%            num_u = num_u +1;
%       end
%    end 
%    L1(t) =sum(L10) /num_u ;
%    L2(t) = 0.5*lammda * norm(D-nn.a{nn.n}', 'fro')^2;
%    L3(t) = 2*alpha*trace(B'*X) + 2*beta*trace(D'*Y);
%    Loss(t)= L1(t) +L2(t) -L3(t) ;  
%   
  
    k=30;
    [hit_sp,mrr_sp] = predict_sp(k, B, D, Test); % testing for sparsity setting(10%)
    [hit_cd, mrr_cd] = predict_cd(nn,cold_item, row, k,cold_rating, B); % testing for cold-start setting
    spmrr(t) = mrr_sp;
    cdmrr(t) = mrr_cd;
    fid=fopen('C:\Users\13184888\Desktop\cae_tkde\dae_pair\sp_acc_result_times.txt','a');
%     fprintf(fid,'sp=%f la=%f  mrr=%f ',sp,lammda,mrr_sp);
%     fprintf(fid,'\n');
    for i=1:k
        fprintf(fid,'%f ',hit_sp(i));
    end
    fprintf(fid,'\n');
    fclose(fid);
    
    fid=fopen('C:\Users\13184888\Desktop\cae_tkde\dae_pair\cd_acc_result_times.txt','a');
%     fprintf(fid,'sp=%f la=%f  mrr=%f ',sp,lammda, mrr_cd);
%     fprintf(fid,'\n');
    for i=1:k
        fprintf(fid,'%f ',hit_cd(i));
    end
    fprintf(fid,'\n');
    fclose(fid);
    
%     akd4=unique(D','rows');
% akb4=unique(B','rows');
    %  ti=toc
end

fid=fopen('C:\Users\13184888\Desktop\cae_tkde\dae_pair\mrr_result_times.txt','a');
   fprintf(fid,'sp=%f ',sp);
      fprintf(fid,'\n');
    for i=1:times
        fprintf(fid,'%f ',spmrr(i));
    end
    fprintf(fid,'\n');
        for i=1:times
        fprintf(fid,'%f ',cdmrr(i));
        end
        fprintf(fid,'\n');
    fclose(fid);
    
  %% plot the training loss of each objective  
% figure;
% plot(1:times,L1, 'g-s');
% hold on
% % plot(1:times,L2, 'b^-');
% plot(1:times,L2/(0.5*lammda), 'b^-');
% hold on
% plot(1:times,L3, 'm-x');
% hold on
% plot(1:times,Loss, 'r-o');
% legend('Loss1','Loss2','Loss3','Loss_total')
%   title('Training loss')
%   xlabel('times')
%   ylabel('loss value')
%   
%  figure;
% plot(1:times,L1, 'g-s');
%  figure;
% plot(1:times,L2/(0.5*lammda), 'b^-');
end

% function [ B ] = Bsub1( Rt, B, D, X, r, N, M, Ia, Is, alpha)
% % B-subproblem
% DD=D*ones(M,1);
% for u=1:N
%     DDa(:,u)=D*Rt(u,:)'; %rx1
%     Da(:,u)=Ia(u)*DDa(:,u);
%     Ds(:,u)=Is(u)*(DD-DDa(:,u)); %rx1
% FLAg=1; step=0;
% %while FLAg
% for k=1:r
%              dda=D(k,:)*Rt(u,:)';
%              da(k)=Ia(u)*dda;
%              dds=D(k,:)*ones(M,1)-dda;
%              ds(k)=Is(u)*dds;
%              Dda=(D(k,:).*Rt(u,:)*D')';
%              dDa=Ia(u)*Dda;            %r*1
%              Dds=D*D(k,:)'-Dda;
%              dDs=Is(u)*Dds;  %r*1
%
%              bu0_bar(k)=(-B(:,u)'*Ds(:,u)+B(k,u)*ds(k))*da(k)-(B(:,u)'*Da(:,u)-B(k,u)*da(k))*ds(k)+(B(:,u)'*dDa-r*da(k))+(B(:,u)'*dDs+r*ds(k))-2*B(k,u);
%              bu_bar(k)=-bu0_bar(k)+alpha*N*X(k,u);
%              if bu_bar(k)~=0
%                  if B(k,u)==sign(bu_bar(k))
%                      fl(k)=0;
%                  else
%                      B(k,u)=sign(bu_bar(k));
%                      fl(k)=1;
%                  end
%              else
%                  continue;
%              end
% %end
%            FLAg=sum(fl);
%           step=step+1;
% end
% end
% end


% function [ D ]=Dsub1(Rt, B, D, Y, r, Ia, Is, N, M, beta, DBN_out, lammda)
% %D-subproblem
% DD=D*ones(M,1);
% for u=1:N
%        DDa(:,u)=D*Rt(u,:)'; %rx1
%        Da(:,u)=Ia(u)*DDa(:,u);
%        Ds(:,u)=Is(u)*(DD-DDa(:,u)); %rx1
%        BDa(u)=B(:,u)'*Da(:,u);
%        BDs(u)=B(:,u)'*Ds(:,u);
% end
% for i=1:M
%        za(i)=Ia'*Rt(:,i);
%        zs(i)=Is'*(ones(N,1)-Rt(:,i));
%
%        flag2=1; step=0;
%        while flag2
%    for k=1:r
%        zba(k)=(Ia'.*B(k,:))*Rt(:,i);
%        zbs(k)=(Is'.*B(k,:))*(ones(N,1)-Rt(:,i));
%        di0_bar(k,:)=B(k,:).*((ones(1,N)-Rt(:,i)').*Is')*(B'*D(:,i)-BDa');
%        di1_bar(k,:)=B(k,:).*(Rt(:,i)'.*Ia')*(B'*D(:,i)-BDs');
%        di_bar3(k)=di0_bar(k,:)+di1_bar(k,:)-(za(i)+zs(i))*D(k,i)+r*zbs(k)-r*zba(k);
%        di_bar(k)=-di_bar3(k)+N*beta*Y(k,i) + N*lammda*DBN_out(i,k)/2;
%        if di_bar(k)~=0
%            if D(k,i)==sign(di_bar(k))
%                fl2(k)=0;
%            else
%               D(k,i)=sign(di_bar(k));
%               fl2(k)=1;
%            end
%            continue;
%        end
%     end
%         flag2=sum(fl2);
%         step=step+1;
% end
% end
%
% end


