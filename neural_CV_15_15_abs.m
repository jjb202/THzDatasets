clc;
clear all;
close all;
%read data change 3d data
% load rawimage; 
% [rowNum,colNum]=size(rawimage);
% 
% 
%     for r=1:15
%          for c=1:15
%         rawimage_250g(r,c)=rawimage(112-8+r,110-8+c);
%          end
%     end
%     
% 
%         absImage=abs(rawimage_250g);
%         [c,l]=max(absImage);
%         angleImage=angle(rawimage_250g);
%    realImage=real(rawimage);
%    imagImage=imag(rawimage);
%   save('fact_facula_15_15.mat', 'rawimage_250g');
%    save(' fact_facula_abs_15_15.mat', 'absImage'); 
%    save('fact_facula_angle_15_15.mat', 'angleImage');
%    save('fact_facula_real_15_15.mat', 'realImage');
%     save('fact_facula_imag_15_15.mat', 'imagImage');
% 
%         figure()
%         imagesc(absImage);
%         title('intensity');
%         figure()       
%         imagesc(angleImage);
%         title('phase');
%       figure()
%       
%      imagesc(realImage);
%       title('real');
%         figure()
%              
%      imagesc(imagImage);
%       title('imag');
train_data=zeros(64,64,1,6400);
train_label=zeros(64,64,6400);
load('fact_facula_15_15.mat')%%%%%15*15

    for m=0:9
       for n=1:640
        %¶ÁÈ¡Ñù±¾
        train_data_R=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));%original image 0-255
        
%          figure()
%          imshow(train_data_R); 
%         train_data_R=double(train_data_R);%%double
         train_data_R= imresize(train_data_R, [64 64],'bilinear');
%          if n==1
%          figure()
%          imshow(train_data_R); 
%          title('original image');
%          end
%          train_data_N= normalization_image(train_data_R)  ;
          train_data_N=im2bw(train_data_R,0.5);%(0,.1)
%          
%          figure()
%          imshow(train_data_N); 
%          title('original image');
%        
%          train_data_B=im2bw(train_data_R,0.5);%(0,.1)
        train_data_B=double(train_data_N);%%double
%         if n==1
%         figure
%          imagesc(train_data_B); 
%         end
        train_data_temp=conv2(train_data_B,rawimage_250g,'same'); 
%         absImage=abs(train_data_temp);
%         absImage_N= normalization_image(absImage)  ;
     
        angleImage=angle(train_data_temp);
%         Image_N= normalization_image(angleImage)  ;
%         absImage_N=abs(train_data_temp);
%         Image_N= normalization_image(absImage_N)  ;
%         realImage=real(train_data_temp);
%         realImage_N= normalization_image(realImage)  ;
%         imagImage=imag(train_data_temp);
%         imagImage_N= normalization_image(imagImage)  ;
%         Image_N=complex(realImage_N,imagImage_N);
%         phaseImage=angle(train_data_temp);
%         Image_N= absImage_N.*exp(phaseImage*1i); 
%         absImage_N=abs(Image_N);

%         phaseImage=angle(train_data_temp);
%         Image_N= absImage_N.*exp(phaseImage*1i); 

%         absImage_P=absImage_N*255;
%   angleImage=angle(train_data_temp);
%    realImage=real(train_data_temp);
%    imagImage=imag(train_data_temp);
%   
% if n==1
%         figure()  
%         imshow(absImage_N);
%         title('intensity image');
%       if m==3
%           aa=1;
%       end
% end
%          figure()
%       imagesc(train_data_B);
%       title('original image');
%      imagesc(angleImage);
%         title('phase');
% if n==100
% figure()
% imshow(realImage_N);
% title('real');
% figure()
% imshow(imagImage_N);
% title('imag');
% end
       
       for k=1
       train_data(:,:,k,m*640+n)=angleImage;
       
       train_label(:,:,m*640+n)=train_data_N;
       end
%         aa(:,:,:)=train_data(:,:,:,1);
       end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train_label=zeros(10,7000); 
% for a=0:9
%     for b=1:700
%         train_label(a+1,a*700+b)=complex(1,1);
%         
%     end
% %     
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   val_data=zeros(64,64,1,1600);
   val_label=zeros(64,64,1600);
% load('facula_8_8.mat')

    for m=0:9
       for n=701:860
        
        train_data_R=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));
%          if n==701
%          figure()
%          imshow(train_data_R); 
%          title('original image1');
%          end
%        
%          figure(1)
%          imshow(train_data); 
%           train_data_B=im2bw(train_data_R,0.5);
%         train_data_B=double(train_data_B);
%         figure(2)
%          imshow(train_data); 
         train_data_R= imresize(train_data_R, [64 64],'bilinear');
         
%           if n==701
%          figure()
%          imshow(train_data_R); 
%          title('original image3');
%          end
%          train_data_N= normalization_image(train_data_R)  ;
         train_data_N=im2bw(train_data_R,0.5);%(0,.1)
%                if n==701
%          figure()
%          imshow(train_data_N); 
%          title('original image3');
%          end
        train_data_B=double(train_data_N);
        train_data_temp=conv2(train_data_B,rawimage_250g,'same');
%         absImage=abs(train_data_temp);
        angleImage=angle(train_data_temp);
%         absImage_N= normalization_image(absImage)  ;
%         realImage=real(train_data_temp);
%         realImage_N= normalization_image(realImage)  ;
%         imagImage=imag(train_data_temp);
%         imagImage_N= normalization_image(imagImage)  ;
%         Image_N=complex(realImage_N,imagImage_N);
%         phaseImage=angle(train_data_temp);
%         Image_N= absImage_N.*exp(phaseImage*1i); 
%         absImage_N=abs(Image_N);
       
%         absImage=abs(train_data_temp);
%         absImage_N= normalization_image(absImage)  ;
%         absImage_P=absImage_N*255;
       
       for k=1
       val_data(:,:,k,m*160+n-700)=angleImage;
       val_label(:,:,m*160+n-700)=train_data_N;
       end
%         aa(:,:,:)=train_data(:,:,:,1);
       end
    end 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     val_label=zeros(10,70); 
%     
% for aa=0:9
%     for bb=1:70
%            val_label(aa+1,aa*70+bb)=complex(1,1);
%     end
%     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_data=zeros(50,50,1,700);
% load('facula_8_8.mat')

%     for m=0:9
%        for n=801:870
%        
%         train_data_R=imread(strcat(num2str(m),'_',num2str(n),'.bmp'));
% %        
% %          figure(1)
% %          imshow(train_data); 
%           train_data_B=im2bw(train_data_R,0.5);
%         train_data_B=double(train_data_B);
% %         figure(2)
% %          imshow(train_data); 
%        train_data_temp=conv2(train_data_B,rawimage,'same');
%        
%        for k=1
%        test_data(:,:,k,m*70+n-800)=train_data_temp;
%        end
% %         aa(:,:,:)=train_data(:,:,:,1);
%        end
%     end
 fileDir = []; % 
savePath = strcat(fileDir,  'dataset_CV_64_15_15_angle.mat'); 

save(savePath, 'train_data', 'train_label','val_data','val_label','test_data'); 

      % save('data.mat','train_data','train_label','val_data')
%          train_data_temp_real=real(train_data_temp(:,:,n));
%           train_data_temp_imag=imag(train_data_temp(:,:,n));
%           train_data_temp_abs=abs(train_data_temp(:,:,n));
%           train_data_temp_angle=angle(train_data_temp(:,:,n));
%         if((m==9)&&(n==30))
%             imshow(train_data); 
%              figure(n)
%        absImage=abs(train_data_temp(:,:,n));
%      imshow(absImage);

             
%              if(m==0)
%                  train_data_zero=train_data_temp;
%                   save('train_data_zero.mat','train_data_zero')
%                   train_data_zero_real(:,:,n)=train_data_temp_real;
%                   save('train_data_zero_real.mat','train_data_zero_real')
%                   train_data_zero_imag(:,:,n)=train_data_temp_imag;
%                   save('train_data_zero_imag.mat','train_data_zero_imag')
%                   train_data_zero_abs(:,:,n)=train_data_temp_abs;
%                   save('train_data_zero_abs.mat','train_data_zero_abs')
%                   train_data_zero_angle(:,:,n)=train_data_temp_angle;
%                   save('train_data_zero_angle.mat','train_data_zero_angle')
%                 if (n==50)
%                   figure 
%                   imshow(train_data);                    
%                   figure 
%                   imshow(train_data_temp_real);
%                   figure 
%                   imshow(train_data_temp_imag);
%                      figure 
%                   imshow(train_data_temp_abs);
%                   figure 
%                   imshow(train_data_temp_angle);
%                 end
%              end
%                 if(m==1)
%                  train_data_one=train_data_temp;
%                   save('train_data_one.mat','train_data_one')
%                     train_data_one_real(:,:,n)=train_data_temp_real;
%                   save('train_data_one_real.mat','train_data_one_real')
%                     train_data_one_imag(:,:,n)=train_data_temp_imag;
%                   save('train_data_one_imag.mat','train_data_one_imag')
%                   train_data_one_abs(:,:,n)=train_data_temp_abs;
%                   save('train_data_one_abs.mat','train_data_one_abs')
%                   train_data_one_angle(:,:,n)=train_data_temp_angle;
%                   save('train_data_one_angle.mat','train_data_one_angle')
% 
%                     if (n==50)
%                            figure 
%                   imshow(train_data);    
% %                 figure 
% %                   imshow(train_data_temp_real);
% %                   figure 
% %                   imshow(train_data_temp_imag);
% %                      figure 
% %                   imshow(train_data_temp_abs);
% %                   figure 
% %                   imshow(train_data_temp_angle);
%                 end
%                 end
%                  if(m==2)
%                  train_data_two=train_data_temp;
%                   save('train_data_two.mat','train_data_two')
%                     train_data_two_real(:,:,n)=train_data_temp_real;
%                   save('train_data_two_real.mat','train_data_two_real')
%                     train_data_two_imag(:,:,n)=train_data_temp_imag;
%                   save('train_data_two_imag.mat','train_data_two_imag')
%                   train_data_two_abs(:,:,n)=train_data_temp_abs;
%                   save('train_data_two_abs.mat','train_data_two_abs')
%                   train_data_two_angle(:,:,n)=train_data_temp_angle;
%                   save('train_data_two_angle.mat','train_data_two_angle')
% 
%                     if (n==50)
%                            figure 
%                   imshow(train_data);    
% %                figure 
% %                   imshow(train_data_temp_real);
% %                   figure 
% %                   imshow(train_data_temp_imag);
% %                      figure 
% %                   imshow(train_data_temp_abs);
% %                   figure 
% %                   imshow(train_data_temp_angle);
%                 end
%                  end
%                if(m==3)
%                   train_data_three=train_data_temp;
%                   save('train_data_three.mat','train_data_three')
%                     train_data_three_real(:,:,n)=train_data_temp_real;
%                   save('train_data_three_real.mat','train_data_three_real')
%                     train_data_three_imag(:,:,n)=train_data_temp_imag;
%                   save('train_data_three_imag.mat','train_data_three_imag')
%                   train_data_three_abs(:,:,n)=train_data_temp_abs;
%                   save('train_data_three_abs.mat','train_data_three_abs')
%                   train_data_three_angle(:,:,n)=train_data_temp_angle;
%                   save('train_data_three_angle.mat','train_data_three_angle')
% 
%                     if (n==50)
%                            figure 
%                   imshow(train_data);    
% %            figure 
% %                   imshow(train_data_temp_real);
% %                   figure 
% %                   imshow(train_data_temp_imag);
% %                      figure 
% %                   imshow(train_data_temp_abs);
% %                   figure 
% %                   imshow(train_data_temp_angle);
%                 end
%                end
%                if(m==4)
%                   train_data_four=train_data_temp;
%                   save('train_data_four.mat','train_data_four')
%                     train_data_four_real(:,:,n)=train_data_temp_real;
%                   save('train_data_four_real.mat','train_data_four_real')
%                     train_data_four_imag(:,:,n)=train_data_temp_imag;
%                   save('train_data_four_imag.mat','train_data_four_imag')
%                   train_data_four_abs(:,:,n)=train_data_temp_abs;
%                   save('train_data_four_abs.mat','train_data_four_abs')
%                   train_data_four_angle(:,:,n)=train_data_temp_angle;
%                   save('train_data_four_angle.mat','train_data_four_angle')
% 
%                     if (n==50)
%                            figure 
%                   imshow(train_data);    
% %                figure 
% %                   imshow(train_data_temp_real);
% %                   figure 
% %                   imshow(train_data_temp_imag);
% %                      figure 
% %                   imshow(train_data_temp_abs);
% %                   figure 
% %                   imshow(train_data_temp_angle);
%                 end
%                end
%                if(m==5)
%                  train_data_five=train_data_temp;
%                   save('train_data_five.mat','train_data_five')
%                     train_data_five_real(:,:,n)=train_data_temp_real;
%                   save('train_data_five_real.mat','train_data_five_real')
%                     train_data_five_imag(:,:,n)=train_data_temp_imag;
%                   save('train_data_five_imag.mat','train_data_five_imag')
%                   train_data_five_abs(:,:,n)=train_data_temp_abs;
%                   save('train_data_five_abs.mat','train_data_five_abs')
%                   train_data_five_angle(:,:,n)=train_data_temp_angle;
%                   save('train_data_five_angle.mat','train_data_five_angle')
% 
%                     if (n==50)
%                            figure 
%                   imshow(train_data);    
% %                   figure 
% %                   imshow(train_data_temp_real);
% %                   figure 
% %                   imshow(train_data_temp_imag);
% %                      figure 
% %                   imshow(train_data_temp_abs);
% %                   figure 
% %                   imshow(train_data_temp_angle);
%                 end
%                end
%                 if(m==6)
%                   train_data_six=train_data_temp;
%                   save('train_data_six.mat','train_data_six')
%                     train_data_six_real(:,:,n)=train_data_temp_real;
%                   save('train_data_six_real.mat','train_data_six_real')
%                     train_data_six_imag(:,:,n)=train_data_temp_imag;
%                   save('train_data_six_imag.mat','train_data_six_imag')
%                            train_data_six_abs(:,:,n)=train_data_temp_abs;
%                   save('train_data_six_abs.mat','train_data_six_abs')
%                   train_data_six_angle(:,:,n)=train_data_temp_angle;
%                   save('train_data_six_angle.mat','train_data_six_angle')
% 
%                     if (n==50)
%                            figure 
%                   imshow(train_data);    
% %           figure 
% %                   imshow(train_data_temp_real);
% %                   figure 
% %                   imshow(train_data_temp_imag);
% %                      figure 
% %                   imshow(train_data_temp_abs);
% %                   figure 
% %                   imshow(train_data_temp_angle);
%                 end
%                 end
%              if(m==7)
%                   train_data_seven=train_data_temp;
%                   save('train_data_seven.mat','train_data_seven')
%                     train_data_seven_real(:,:,n)=train_data_temp_real;
%                   save('train_data_seven_real.mat','train_data_seven_real')
%                     train_data_seven_imag(:,:,n)=train_data_temp_imag;
%                   save('train_data_seven_imag.mat','train_data_seven_imag')
%                        train_data_seven_abs(:,:,n)=train_data_temp_abs;
%                   save('train_data_seven_abs.mat','train_data_seven_abs')
%                   train_data_seven_angle(:,:,n)=train_data_temp_angle;
%                   save('train_data_seven_angle.mat','train_data_seven_angle')
%          
%                     if (n==50)
%                            figure 
%                   imshow(train_data);    
% %                      figure 
% %                   imshow(train_data_temp_real);
% %                   figure 
% %                   imshow(train_data_temp_imag);
% %                      figure 
% %                   imshow(train_data_temp_abs);
% %                   figure 
% %                   imshow(train_data_temp_angle);
%                 end
%              end
%                 if(m==8)
%                  train_data_eight=train_data_temp;
%                   save('train_data_eight.mat','train_data_eight')
%                     train_data_eight_real(:,:,n)=train_data_temp_real;
%                   save('train_data_eight_real.mat','train_data_eight_real')
%                     train_data_eight_imag(:,:,n)=train_data_temp_imag;
%                   save('train_data_eight_imag.mat','train_data_eight_imag')
%                   train_data_eight_abs(:,:,n)=train_data_temp_abs;
%                   save('train_data_eight_abs.mat','train_data_eight_abs')
%                   train_data_eight_angle(:,:,n)=train_data_temp_angle;
%                   save('train_data_eight_angle.mat','train_data_eight_angle')
% 
%                     if (n==50)
%                            figure 
%                   imshow(train_data);    
% %                   figure 
% %                   imshow(train_data_temp_real);
% %                   figure 
% %                   imshow(train_data_temp_imag);
% %                      figure 
% %                   imshow(train_data_temp_abs);
% %                   figure 
% %                   imshow(train_data_temp_angle);
%                 end
%                 end
%                  if(m==9)
%                   train_data_nine=train_data_temp;
%                   save('train_data_nine.mat','train_data_nine')
%                     train_data_nine_real(:,:,n)=train_data_temp_real;
%                   save('train_data_nine_real.mat','train_data_nine_real')
%                     train_data_nine_imag(:,:,n)=train_data_temp_imag;
%                   save('train_data_nine_imag.mat','train_data_nine_imag')
%              train_data_nine_abs(:,:,n)=train_data_temp_abs;
%                   save('train_data_nine_abs.mat','train_data_nine_abs')
%                   train_data_nine_angle(:,:,n)=train_data_temp_angle;
%                   save('train_data_nine_angle.mat','train_data_nine_angle')
% 
% 
%                     if (n==50)
%                            figure 
%                   imshow(train_data);    
% %                     figure 
% %                   imshow(train_data_temp_real);
% %                   figure 
% %                   imshow(train_data_temp_imag);
% %                      figure 
% %                   imshow(train_data_temp_abs);
% %                   figure 
% %                   imshow(train_data_temp_angle);
%                 end
%              end
%    
%              end
%     end



    