clear
for s=1:9
sp=0.1*s;
lammda=10*(2+s); 
main_sae_pair;
spmrr(s)=mrr_sp;
cdmrr(s)=mrr_cd;
end
fid=fopen('C:\Users\13184888\Desktop\cae_tkde\dae_pair\mrr_result.txt','a');
fprintf(fid,'lammda=%f ', lammda);
fprintf(fid,'\n');
for i=1:9
    fprintf(fid,'%f ',spmrr(i));
end
fprintf(fid,'\n');
for i=1:9
    fprintf(fid,'%f ',cdmrr(i));
end
fprintf(fid,'\n');
fclose(fid);
clear