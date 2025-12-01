function [NofF,iname1,iname2,name,N,tau,Epsmin,Epsmax,EN,Estep,dmin,dmax,deltad,dM,tbegin,tend,tstep,Cal1,Cal2,headerA,time,dataA,dataB,new] = analyse_HT7_inp_file(Val)


Val = Val.';
Val{1} = str2num(Val{1});
for i=5:19 
Val{i} = str2num(Val{i});
end

NofF = Val{1,1};
iname1 = Val{2,1};
iname2 = Val{3,1};
name = Val{4,1};
 N = Val{5,1}; % Data length to be processed
tau = Val{6,1}; % number of data to be skipped (delay for phase space)
Epsmin = Val{7,1}; % minimum log2(epsilon)
Epsmax = Val{8,1}; % maximum log2(epsilon)
EN = Val{9,1}; % number of epsilons
Estep = Val{10,1}; % step of epsilons
dmin = Val{11,1};  % minimun dimensions
dmax = Val{12,1};  % maximun dimensions
deltad = Val{13,1};  % maximun dimensions
dM=Val{14,1};      %maximum embedded dimension
tbegin=Val{15,1};   %time begin
tend=Val{16,1};      %time end
tstep=Val{17,1};     %time step
Cal1=Val{18,1};      %calibration factor for signal A
Cal2=Val{19,1};      %calibration factor for signal B


  sinfo= sprintf('%s.par',name);
      [pathst,nam,ext] = fileparts(sinfo);
      new = sprintf('%s',nam,ext);
     fid=fopen(new,'wt');
%%write iuput file information to .par file and close, 'wt' means overwrite
%%the file.

 fprintf(fid,'Number of file pairs       :     %d\n',NofF);
 fprintf(fid,'First file                 :     %s\n',iname1);
 fprintf(fid,'Second file                :     %s\n',iname2);
 fprintf(fid,'File name                  :     %s\n',name);
 fprintf(fid,'data length                :     %d\n',N);
 fprintf(fid,'Delay tau                  :     %d\n',tau); 
 fprintf(fid,'Minimum of log2(epsilon)   :     %d\n',Epsmin);
 fprintf(fid,'Maximum of log2(epsilon)   :     %d\n',Epsmax);
 fprintf(fid,'Number of epsilon          :     %d\n',EN);
 fprintf(fid,'Epsilon step               :     %d\n',Estep);
 fprintf(fid,'Minimum dimension          :     %d\n',dmin);      
 fprintf(fid,'Maximum dimension          :     %d\n',dmax);       
 fprintf(fid,'Delta d                    :     %d\n',deltad);        
 fprintf(fid,'dM                         :     %d\n',dM); 
 fprintf(fid,'time begin(ms)             :     %d\n',tbegin); 
 fprintf(fid,'time end(ms)               :     %d\n',tend);  
 fprintf(fid,'tstep(ms)                  :     %d\n',tstep);
 fprintf(fid,'Cal1                       :     %d\n',Cal1);
 fprintf(fid,'Cal2                       :     %d\n\n\n',Cal2);
 
 fclose(fid);
 [headerA,time,dataA,dataB] = childf(iname1,iname2); 
 dataA=dataA*Cal1;
 dataB=dataB*Cal2;
end
 function [headerA,time,dataA,dataB] = childf(iname1,iname2)

                fidA = fopen(iname1);
       a = textscan(fidA,'%s','Delimiter','\n');
         headerA = a{1}(1:7);  %contains the first 7 lines of A.dat 
         fclose(fidA);
         
           fidB = fopen(iname2);
       b = textscan(fidB,'%s','Delimiter','\n');
         headerB = b{1}(1:7);  %contains the first 7 lines of A.dat 
         fclose(fidB);
         
       Start_line=6;
       MatLab_version=findstr(version,'R2019b'); %% Start_line=7 for MatLab R2019b (Xiao's office computer)
       if (MatLab_version~=0) 
           Start_line=Start_line+1; 
       end   
         Anpq= readtable(iname1,'Delimiter',' ','headerLines',Start_line);
         Bnpq= readtable(iname2,'Delimiter',' ','headerLines',Start_line);
         
         time=Anpq{:,1};
         dataA=Anpq{:,3};
         dataB=Bnpq{:,3};
    end

 
