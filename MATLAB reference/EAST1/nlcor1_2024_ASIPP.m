% NLCOR1.m
% Non-linear correlation coefficient (version 1)
% by W. Ding, USTC, China
% Modified by C. Xiao, Univ. of Saskatchewan, Canada (Mar.19, 1997)
% Converted to MatLab by Vineet Kumar DAS (MITACS Intern),  Univ. of Saskatchewan, Canada (August, 2022)
% Input file: infile (input from keyboard, e.g., nlcor1.inp)
% Output files:  infile.cor
%  NAMEn.dis (n is the maximum dimension specified in infile)

% 2024 Corrected a problem when reading data from the dispersion data file
% befor plotting. the correct statement is  
% extract = readtable(fplot,'Format','%f  %f  %f  %f  %f','headerLines',Start_line) 
% Corresponding to the the format when the data was written.


%% modified and made it work by Vineet (Mitacs summer sturdent) May-July, 2022
%% tested by C. Xiao for the input file test_cx.inp 
%%    and compared with the run NC_ASIPP.EXE (under command prompt) for the input file test_cx.par
%% added plotting and saving parts to the original program
%% Nonliner cross correlation coefficients are saved in the file: test_cx.par
%% plotting data are saved in the following files: test_cx1.dat, test_cxi.dat, ... test_cx9.dat

clear all;
close all;
pr=('Enter the parameter input file (*.inp, e.g., test_AB.inp or test_BA.inp)');
fn_inp = input(pr,'s');
fid1 = fopen(fn_inp);
% Read all lines & collect in cell array
Val = textscan(fid1,'%s','Delimiter','\n');
fclose(fid1);
Val = Val{1};
Val = Val.';

%calling
%function(analyse_HT7_inp_file)---------------------------------------------------------------------------------------------------------------------------------------
 [NofF,iname1,iname2,name,N,tau,Epsmin,Epsmax,EN,Estep,dmin,dmax,deltad,dM,tbegin,tend,tstep,Cal1,Cal2,headerA,time,dataA,dataB,new] = analyse_HT7_inp_file(Val);
%---------------------------------------------------------------------------------------------------------------------------------------------------------- 

pr1='Do you want to continue <y/n>?';
  str1 = input(pr1,'s');
  if (strcmp(str1,'y')||strcmp(str1,'Y'))
   
         idx1 = strfind(headerA{6},'=');
         idx2 = strfind(headerA{6},' ');
         p=headerA{6}((idx1+1):(idx2-1));
         dt = str2num(p);
         format longg
         dt;
         
%--------------------------------------------------------------------------        
  
       AN=dataA;  % dataA from the input function
       BN=dataB;  % dataB from the input function

       I_begin=(1000.0*tbegin);
       I_end=(1000.0*tend);
       I_step=(1000.0*tstep);
       
       
       [m,n]=size(dataA);
       t=0:(m-1);
       t=dt*t;
       t=t.';
   
        
         
       for I_time=I_begin:I_step:I_end  %test_cx.inp 100:10:105 did only t=100
           
           AN11=AN((I_time+1):(I_time+N));
           BN11=BN((I_time+1):(I_time+N));
   
          ANMAX=max(AN11);
          BNMAX=max(BN11);
           
          ANMIN=min(AN11);
          BNMIN=min(BN11);
           
          ANMAX=ANMAX-ANMIN;    
          BNMAX=BNMAX-BNMIN; 
          fprintf('I_time         t(ms)                AN11                          BN11\n');
          for o=1:5
          fprintf('%d        %E        %E                %E\n',(o+I_time),((o+I_time)*dt*1000),AN11(o),BN11(o));
          end
          
          
     if((ANMAX~=0)&&(BNMAX~=0))
        AN1=AN11/ANMAX;
        BN1=BN11/BNMAX;
     end
     
      fprintf('ANMAX   %f\n', ANMAX); 
      fprintf('BNMAX   %f\n', BNMAX); 
      
      
         A=2.^Epsmin;
         B=Epsmax-Epsmin;
         B=B*log(2.0)/EN;
         
         for I=1:Estep:EN
            EPS(I)=A*exp(B*I);
         end
         
         
         for k=1:Estep:EN
             res(k)=log2(EPS(1,k));
              
         end
        
        if(dM<dmax)
            dM=dmax;
        end
            
      fprintf('%s        %s          %s\n',iname1,iname2,name);
      
       fid=fopen(new,'at');  %opens test_cx.par file for appending the contents. 'at' means append.

       % for D=dmax:deltad:dmax    %last dimension only
            for D=dmin:deltad:dmax %use this to get all dimension (real)
           M=tau*(D-1);
        fprintf('Dimension=%d\n',D);
       sdat = sprintf('%s%d.dat',name,D);
       
       
       fprintf(fid,'Dimension=%d \n',D);    %writes to the existing  test_cx.par file
       fprintf(fid,'File X: %s            File Y:  %s\n',iname1,iname2);  %writes in test_cx.par file
       
        % SUM for fixed epsilon and dimensionality
        
         for i=1:4
            for L=1:Estep:EN 
              SUM1(i,L)=0.0;
              SUM2(i,L)=0.0;
            end
         end

           
          for i=1:N-M-1
              for j=i+1:N-M
                %Distancae between two points for x,y
                 DX=0.0;
                 DY=0.0;
                for k=0:tau:M
                   DX=DX+(AN1(i+k)-AN1(j+k))*(AN1(i+k)-AN1(j+k));
                    DY=DY+(BN1(i+k)-BN1(j+k))*(BN1(i+k)-BN1(j+k));
                end
           
                    DX=DX/(dM);
                    DXSQRT=sqrt(DX);
                    DY=DY/(dM);
                    DYSQRT=sqrt(DY);  
                
                for L=EN:-Estep:1
                   if(EPS(L)>DXSQRT)
                       SUM1(1,L)=SUM1(1,L)+DX;    % SUM for Sigma_XX
                       SUM2(1,L)=SUM2(1,L)+1.0;
                       SUM1(2,L)=SUM1(2,L)+DY;    % SUM for Sigma_XY
                       SUM2(2,L)=SUM2(2,L)+1.0;
                   else
                      break;
                   end
                end
               
                for L1=EN:-Estep:1
                   if(EPS(L1)>DYSQRT)
                        SUM1(3,L1)=SUM1(3,L1)+DY;    % SUM for Sigma_YY
                        SUM2(3,L1)=SUM2(3,L1)+1.0;
                        SUM1(4,L1)=SUM1(4,L1)+DX;    % SUM for Sigma_YX
                        SUM2(4,L1)=SUM2(4,L1)+1.0; 
                   else
                      break;
                   end               
                end
                       
           end
        end
      
    for i=1:4
       for j=1:Estep:EN
          if(SUM2(i,j)>0.5)
              SUM1(i,j)=sqrt(SUM1(i,j)/SUM2(i,j));
          end
       end
    end
    
    
  %NORMALIZATION
  %FIND maximum
  
     SUM1MAX=1:1:4;
    for i=1:4
       SUM1MAX(i)=max(SUM1(i,:));
    end
    
  
  %Normalise to 1
  [pathstrdat,namedat,extdat] = fileparts(sdat);
      newdat = sprintf('%s',namedat,extdat);
      fiddat=fopen(newdat,'wt');  %opens test_cx1.dat or ... or .test_cx9.dat for overwrite for each Dimension
  fprintf(fiddat,'log(E)     XX        XY          YY             YX\n');
  for L=1:Estep:EN
  fprintf(fiddat,'%f  ',res(L));    
     for i=1:4
        if(SUM1MAX(i)>0.0)
           SUM1(i,L)=SUM1(i,L)/SUM1MAX(i); 
        end
      fprintf(fiddat,'%f  ',SUM1(i,L));  
     end
     fprintf(fiddat,'\n');
  end
  
  fclose(fiddat);   % opens test_cx1.dat or ... or .test_cx9.dat 
  
 % Find the epsilon at which log2(epsilon)=0.9 
       
   fprintf(fid,'epsilon at which log2(epsilon)=0.9-------------\n'); 
   fprintf(fid,'XX             XY             YY             YX\n');
  for i=1:4 
     if(SUM1(i,EN)<0.9)
        k1=EN;
        k2=EN-1;
     else
         for L=1:Estep:EN
            if(SUM1(i,L)>0.9)
                k1=L-1;
                k2=L;
               if(L==1)
                   k1=2;
               end
               break;
            end
         end
         fprintf(fid,'%f        ',EPS(k1));
     end
    
 
    if(SUM1(i,k2)~=SUM1(i,k1))
        Eps4(i)=EPS(k1)+(EPS(k2)-EPS(k1))*(0.9-SUM1(i,k1))/(SUM1(i,k2)-SUM1(i,k1));
     else
         Eps4(i)=EPS(k1);
     end
  end
  fprintf(fid,'\n');
 
  
  %Output Data
fprintf(fid,'EpsXX= %E             EpsXY=  %E\n',Eps4(1),Eps4(2));
fprintf(fid,'EpsYY= %E             EpsYX=  %E\n',Eps4(3),Eps4(4));
fprintf(fid,'Gxy=EXY/sqrt(EXX EYY)=    %E\n', Eps4(2)/sqrt(Eps4(1)*Eps4(3)));
fprintf(fid,'Gyx=EYX/sqrt(EXX EYY)=    %E\n', Eps4(4)/sqrt(Eps4(1)*Eps4(3)));
fprintf(fid,'Sxy=(EYX-EXY)/(EXY+EYX)=  %E\n', (Eps4(4)-Eps4(2))/(Eps4(2)+Eps4(4)));
fprintf(fid,'\n');

      end %end of dimension(D)
   fclose(fid);  %cloes test_cx.par file
       end  %I_time
  
%plotting-------------------------------------------------------------------
  
  j=1;
  legendInfo1 = cell(length(dmin:deltad:dmax),1);
  legendInfo2 = cell(length(dmin:deltad:dmax),1);
  
  fig=figure(1);
  
  for D=dmin:deltad:dmax
     fplot = sprintf('%s%d.dat',name,D);
     XY = sprintf('XY_%d',D);
     YX = sprintf('YX_%d',D);
     extract =sprintf('extract_%d',D);
     
     Start_line=0; % default for all other MatLab versions
       MatLab_version=findstr(version,'R2019b'); %Xiao office computer
       if (MatLab_version~=0) 
           Start_line=Start_line+1;  
       end
     extract = readtable(fplot,'Format','%f  %f  %f  %f  %f','headerLines',Start_line);
     %extract = readtable(fplot,'Delimiter',' ','headerLines',Start_line); %% Start_line=1 for MatLab R2019b (Xiao's office computer 
     %% or extract = readtable(fplot,'Delimiter',' ','headerLines',0);
     res=extract{:,1};
     XY=extract{:,3};
     YX=extract{:,5};
     ax1 = subplot(1,2,1);
     plot(res,XY,'-o','DisplayName',['M = ',num2str(D)]);
     ylabel('sigma XY(E)');
     xlabel('logE (base 2)');
     legendInfo1{j}=['M = ',num2str(D)];
     hold on;
     grid on;

     ax2 = subplot(1,2,2);
     plot(res,YX,'-o', 'DisplayName',['M = ',num2str(D)]);
     ylabel('sigma YX(E)');
     xlabel('logE (base 2)');
     legendInfo2{j}=['M = ',num2str(D)];
     hold on;
     grid on;
     j=j+1;
  end  
legend(ax1,legendInfo1);
legend(ax2,legendInfo2);
hold off;
figname=strcat(name,'_figure.mfig');  %name for the figure
saveas(fig,figname);   %saves the figure(test_cx_figure.fig)
  end  % this the end of if statement
%----------------------------------------------------------------------------
% %STOP
% %END



