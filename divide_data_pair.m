function [cold_item,cold_rating,row,content,Train,Test] = divide_data_pair(rating,content,optisize,sp)
rating = rating./5;
% RI=rating>3/5;  % for positive rating: preference means rating>3
RI=rating>0;  % for implicit feedback: preference means rating>0
cold=find(sum(RI)<5);  % cold start items
cold_rating=RI(:,cold); % testing dataset for cold start setting
RI(:,cold)=[]; % training and testing dataset for sparsity settings
row=find(sum(cold_rating,2)>1)';  % choose users with ratings in cold start setting
cold_item=content(cold,:); %content tf-idf features for cold start items
content(cold,:)=[]; % content traing dataset
[~,M]=size(RI);
rr=mod(M,optisize);
if rr~=0
    RI(:,1:rr)=[];
    content(1:rr,:)=[];
end
[N,M]=size(RI);

% randomly choose sp = 10% ratings from rating dataset as training data.
% The rest as testing data
Train = zeros(N,M);
Test = zeros(N,M);
id_nz = find(RI);

% id_train=randperm(length(id_nz),ceil(length(id_nz)*sp));

rand_id = randperm(length(id_nz));
rand('state',0);
id_train = id_nz(rand_id(1:ceil(length(id_nz)*sp)));
% id_test = id_nz(rand_id(1:ceil(length(id_nz)*0.1)));
% id_train =  id_nz(rand_id(ceil(length(id_nz)*0.1)+1 : ceil(length(id_nz)*(0.1+sp))));

id_test = setdiff(id_nz,id_train);
% id_t = setdiff(id_nz,id_test);
% rand_idt = randperm(length(id_t));
% id_train = id_t(rand_idt(1:floor(length(id_t)*sp)));

Train(id_train)=1;
Test(id_test)=1;
Train=Train>0;
Test=Test>0;
Train=sparse(Train);
Test=sparse(Test);

% for j=1:M
%     [I,J,Val] = find(RI(:,j));
%     rand('state',0);
%     [test,train] = crossvalind('LeaveMOut',nnz(RI(:,j)),ceil(nnz(RI(:,j))*sp));
%     Train(:,j)=sparse(I(train), J(train), Val(train),N,1);
%     Test(:,j)=sparse(I(test),J(test), Val(test), N,1);
% end

end