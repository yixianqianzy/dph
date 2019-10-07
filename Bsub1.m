function [ B ] = Bsub1( Rt, B, D, X, r, N, M, Ia, Is, alpha)
% B-subproblem
% Rt=Train; 
DD=D*ones(M,1);
for u=1:N
    DDa(:,u)=D*Rt(u,:)'; %rx1
    Da(:,u)=Ia(u)*DDa(:,u);
    Ds(:,u)=Is(u)*(DD-DDa(:,u)); %rx1
FLAg=1; step=0;
while FLAg
for k=1:r
             dda=D(k,:)*Rt(u,:)';
             da(k)=Ia(u)*dda;
             dds=D(k,:)*ones(M,1)-dda;
             ds(k)=Is(u)*dds;
             Dda=(D(k,:).*Rt(u,:)*D')';
             dDa=Ia(u)*Dda;            %r*1
             Dds=D*D(k,:)'-Dda;
             dDs=Is(u)*Dds;  %r*1
             
             bu0_bar(k)=(-B(:,u)'*Ds(:,u)+B(k,u)*ds(k))*da(k)-(B(:,u)'*Da(:,u)-B(k,u)*da(k))*ds(k)+(B(:,u)'*dDa-r*da(k))+(B(:,u)'*dDs+r*ds(k))-2*B(k,u);
             bu_bar(k)=-bu0_bar(k)+alpha*N*X(k,u);
             if bu_bar(k)~=0
                 if B(k,u)==sign(bu_bar(k))
                     fl(k)=0; 
                 else
                     B(k,u)=sign(bu_bar(k));
                     fl(k)=1; 
                 end
             else
                 continue;
             end
end
           FLAg=sum(fl);
          step=step+1;
end
end
end