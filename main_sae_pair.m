

% addpath(genpath('C:\Users\13184888\Desktop\cae_tkde\dae_pair\wsdm_pair'))
load rating; % user-item interaction info
load content; %content info of items

rand('state',0);
perm_idx1 = randperm(size(rating,1)); %user number
rand('state',0);
perm_idx2 = randperm(size(rating,2)); %item number

user=perm_idx1(1:10000);
item=perm_idx2(1:10000);
rating=rating(user,item);
content=content(item,:);

optisize=100; 
opts.batchsize = optisize;
opts.numepochs = 50;
% opts.momentum = 0;
% opts.alpha     =   1;


% sp=0.4; % sparsity level: 10%
r=30; % hash code dimension 
alpha=2*10^-5;
beta=10^-4;
% lammda=50; 
k=30; % top-k ranking list
times = 10; 
[cold_item,cold_rating,row,content,Train,Test]=divide_data_pair(rating,content,optisize,sp);

del = all(content==0,1);
content(:,del) = [];
[a,b]=size(content);
% m=max(content);
% content=content./(ones(a,1)*m);
m=max(content,[], 2);    % return the max value of each row
content=content./(m*ones(1,b));  % regularize each row by dividing the max value. It represents the frequency of each word
sizes = [b 200]; % the size of the first layer for sae 
cold_item(:, del) = [];

m1=max(cold_item,[], 2);    % return the max value of each row
cold_item=cold_item./(m1*ones(1,b));

[B, D, nn]= train_sae_pair(sp, content,opts, sizes, Train, Test, cold_item, row, cold_rating, r, alpha, beta, lammda, times);
[hit_sp,mrr_sp] = predict_sp(k, B, D, Test); % testing for sparsity setting(10%)
[hit_cd, mrr_cd] = predict_cd(nn,cold_item, row, k,cold_rating, B);% testing for cold-start setting

% save('C:\Users\13184888\Desktop\cae_tkde\dae_pair\B2','B');
% save('C:\Users\13184888\Desktop\cae_tkde\dae_pair\D2','D');
% save('C:\Users\13184888\Desktop\cae_tkde\dae_pair\nn2','nn');

fid=fopen('C:\Users\13184888\Desktop\cae_tkde\dae_pair\sp_acc_result.txt','a');
fprintf(fid,'lammda=%f ',lammda);
fprintf(fid,'\n');
for i=1:k
    fprintf(fid,'%f ',hit_sp(i));
end
fprintf(fid,'\n');
fclose(fid);

fid=fopen('C:\Users\13184888\Desktop\cae_tkde\dae_pair\cd_acc_result.txt','a');
fprintf(fid,' lammda=%f ',lammda);
fprintf(fid,'\n');
for i=1:k
    fprintf(fid,'%f ',hit_cd(i));
end
fprintf(fid,'\n');
fclose(fid);










