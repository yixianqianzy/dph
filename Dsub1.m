function [ D ]=Dsub1(Rt, B, D, Y, r, Ia, Is, N, M, beta, DBN_out, lammda)
%D-subproblem
DD=D*ones(M,1);
for u=1:N
    DDa(:,u)=D*Rt(u,:)'; %rx1
    Da(:,u)=Ia(u)*DDa(:,u);
    Ds(:,u)=Is(u)*(DD-DDa(:,u)); %rx1
    BDa(u)=B(:,u)'*Da(:,u);
    BDs(u)=B(:,u)'*Ds(:,u);
end
for i=1:M
    za(i)=Ia'*Rt(:,i);
    zs(i)=Is'*(ones(N,1)-Rt(:,i));
    
    flag2=1; step=0;
    while flag2
        for k=1:r
            zba(k)=(Ia'.*B(k,:))*Rt(:,i);
            zbs(k)=(Is'.*B(k,:))*(ones(N,1)-Rt(:,i));
            di0_bar(k,:)=B(k,:).*((ones(1,N)-Rt(:,i)').*Is')*(B'*D(:,i)-BDa');
            di1_bar(k,:)=B(k,:).*(Rt(:,i)'.*Ia')*(B'*D(:,i)-BDs');
            di_bar3(k)=di0_bar(k,:)+di1_bar(k,:)-(za(i)+zs(i))*D(k,i)+r*zbs(k)-r*zba(k);
            di_bar(k)=-di_bar3(k)+N*beta*Y(k,i) + N*lammda*DBN_out(i,k)/(2*M);
            if di_bar(k)~=0
                if D(k,i)==sign(di_bar(k))
                    fl2(k)=0;
                else
                    D(k,i)=sign(di_bar(k));
                    fl2(k)=1;
                end
                continue;
            end
        end
        flag2=sum(fl2);
        step=step+1;
    end
end

end
