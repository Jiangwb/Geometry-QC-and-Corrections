%% use elevation information when calculate offset
clear; close all; clc;

%% read seismic data
shot_num = 300; % shot number
trace_num = 96; % trace number per shot
rec_spacing = 100; % receiver spacing, unit=m
shot_spacing = 40; % shot spacing, unit=m
dt=6; % unit=ms 
[Data_2d,SegyTraceHeaders,SegyHeader] = ReadSegy('../data/2007_merge_sk_crg_96trace.sgy'); Nt=168;
disp('read data end');
% read header info
% sourceX=cat(1,SegyTraceHeaders.SourceX);
% sourceY=cat(1,SegyTraceHeaders.SourceY);
% GroupX=cat(1,SegyTraceHeaders.GroupX);
% GroupY=cat(1,SegyTraceHeaders.GroupY);
% shotID=cat(1,SegyTraceHeaders.FieldRecord);
% source_elevation=cat(1,SegyTraceHeaders.SourceSurfaceElevation);
% Group_elevation =cat(1,SegyTraceHeaders.ReceiverGroupElevation);
%
GroupX=cat(1,SegyTraceHeaders.SourceX);
GroupY=cat(1,SegyTraceHeaders.SourceY);
sourceX=cat(1,SegyTraceHeaders.GroupX);
sourceY=cat(1,SegyTraceHeaders.GroupY);
shotID=cat(1,SegyTraceHeaders.FieldRecord);
Group_elevation=cat(1,SegyTraceHeaders.SourceSurfaceElevation);
source_elevation =cat(1,SegyTraceHeaders.ReceiverGroupElevation);
%% calculate how many traces per shot
[M,N] = size(Data_2d);
tmp=shotID(1);
itmp=0;ishot=1;
trace_num_pershot=zeros(shot_num,1);
for itrace=1:N
    if(shotID(itrace)~=tmp)
        trace_num_pershot(ishot)=itmp;
        itmp=1;ishot=ishot+1;
        tmp=shotID(itrace);
    else
        itmp=itmp+1;
    end
end
trace_num_pershot(shot_num)=N-sum(trace_num_pershot(1:shot_num-1)); % last shot
%% convert Data_2d to Data_3d
Data_3d=zeros(M,max(trace_num_pershot),shot_num);
for ishot=1:shot_num
    for irec=1:trace_num_pershot(ishot)
        if (ishot==1)
            Data_3d(:,irec,ishot)=Data_2d(:,irec);
        else
            Data_3d(:,irec,ishot)=Data_2d(:,irec+sum(trace_num_pershot(1:ishot-1)));
        end      
    end
end
%% 
for ishot=1:shot_num
    if (ishot==1)
        shotX(ishot)=sourceX(1)/100; % unit is m
        shotY(ishot)=sourceY(1)/100; % unit is m
        shotZ(ishot)=-source_elevation(1)/100; % unit is m
        for irec=1:trace_num_pershot(ishot)
            recX(ishot,irec)=GroupX(irec)/100;
            recY(ishot,irec)=GroupY(irec)/100;
            recZ(ishot,irec)=-Group_elevation(irec)/100;
        end
    else
        shotX(ishot)=sourceX(sum(trace_num_pershot(1:ishot-1))+1)/100; % unit is m
        shotY(ishot)=sourceY(sum(trace_num_pershot(1:ishot-1))+1)/100; % unit is m
        shotZ(ishot)=-source_elevation(sum(trace_num_pershot(1:ishot-1))+1)/100; % unit is m

        for irec=1:trace_num_pershot(ishot)
            recX(ishot,irec)=GroupX(sum(trace_num_pershot(1:ishot-1))+irec)/100;
            recY(ishot,irec)=GroupY(sum(trace_num_pershot(1:ishot-1))+irec)/100;
            recZ(ishot,irec)=-Group_elevation(sum(trace_num_pershot(1:ishot-1))+irec)/100;
        end
    end
end
    
%% clear memory
%clear SegyTraceHeaders SegyHeader sourceX sourceY GroupX GroupY
%% sort every shot according offset
Data_3d_sort=zeros(Nt,trace_num,shot_num);
dist=10000*ones(shot_num,max(trace_num_pershot));
dist1=10000*ones(shot_num,max(trace_num_pershot));
% dist=zeros(shot_num,max(trace_num_pershot));
% dist1=zeros(shot_num,max(trace_num_pershot));
for ishot=1:shot_num
    for irec=1:trace_num_pershot(ishot)
        dist(ishot,irec)=sqrt((shotX(ishot)-recX(ishot,irec)).^2+(shotY(ishot)-recY(ishot,irec)).^2+(shotZ(ishot)-recZ(ishot,irec)).^2);
    end
    dist1(ishot,:)=dist(ishot,:);
    
    I=zeros(1,max(trace_num_pershot));
    [dist_sort(ishot,:),I]=sort(dist(ishot,:));
    % sort Data_3d by offset. Select first 64 traces
    for itrace=1:trace_num
        Data_3d_sort(:,itrace,ishot)=Data_3d(1:Nt,I(itrace),ishot);
    end
end

%% create label data
shot_label_num=0;
% correct data, label is 1. 
for ishot1=1:shot_num
    shot_label1(:,:,ishot1)=Data_3d_sort(:,:,ishot1);
end
label1=zeros(2,ishot1);
label1(2,1:ishot1)=1;

% create incorrect data, label is 0.
max_random_number=5; % There are at most 5 receivers with wrong location every shot gather.
label2=zeros(2,shot_num);
Data_3d_random=Data_3d;
% add space shift to the shotX, shotY, recX and recY
shotX_shift=shotX;
shotY_shift=shotY;
recX_shift=recX;
recY_shift=recY;
for ishot=1:shot_num
    % add space shift to source position
%     shot_space_shiftX(ishot)=2*(rand(1,1)-0.5)*shot_spacing;  % range from -1*shot_spacing to 1*shot_spacing
%     shot_space_shiftY(ishot)=2*(rand(1,1)-0.5)*shot_spacing;  % range from -1*shot_spacing to 1*shot_spacing
% 	while( shot_space_shiftX(ishot)>=-0.2*shot_spacing && shot_space_shiftX(ishot)<=0.2*shot_spacing )
%         shot_space_shiftX(ishot)=2*(rand(1,1)-0.5)*shot_spacing;
%     end
% 
% 	while( shot_space_shiftY(ishot)>=-0.2*shot_spacing && shot_space_shiftY(ishot)<=0.2*shot_spacing )
%         shot_space_shiftY(ishot)=2*(rand(1,1)-0.5)*shot_spacing;
%     end

    % load space shift
    load('../data/rec_space_shiftX');
    load('../data/rec_space_shiftY');   
    
    shotX_shift(ishot)=shotX(ishot)+shot_space_shiftX(ishot);
    shotY_shift(ishot)=shotY(ishot)+shot_space_shiftY(ishot);    
    
    random_number=randperm(max_random_number,1); % create a random number that range from 1 to max_random_number (5).
    rec=randperm(trace_num,random_number); % select random_number receivers to change the location
%     if random_number==1 && min(rec)>=10
%         rec=randperm(10,random_number);
%     end
%     if min(rec)>=20
%         rec=randperm(20,random_number);
%     end
    
% 	for irec=1:trace_num_pershot(ishot)
%         for irec1=1:random_number
%             if (irec==rec(irec1))      
%                 %space_shiftX=20*(rand(1,1)-0.5)*rec_spacing;  % range from -10*rec_spacing to 10*rec_spacing
%                 %space_shiftY=20*(rand(1,1)-0.5)*rec_spacing;  % range from -10*rec_spacing to 10*rec_spacing
%                 %space_shiftX=10*(rand(1,1)-0.5)*rec_spacing;  % range from -5*rec_spacing to 5*rec_spacing
%                 %space_shiftY=10*(rand(1,1)-0.5)*rec_spacing;  % range from -5*rec_spacing to 5*rec_spacing                
%                 %space_shiftX=6*(rand(1,1)-0.5)*rec_spacing;  % range from -3*rec_spacing to 3*rec_spacing
%                 %space_shiftY=6*(rand(1,1)-0.5)*rec_spacing;  % range from -3*rec_spacing to 3*rec_spacing
%                 space_shiftX=2*(rand(1,1)-0.5)*rec_spacing;  % range from -rec_spacing to rec_spacing
%                 space_shiftY=2*(rand(1,1)-0.5)*rec_spacing;  % range from -rec_spacing to rec_spacing
%                 %space_shift=0*(rand(1,1)-0.5)*rec_spacing;  % no space shift, just for debugging
%                 if space_shiftX<0.2*rec_spacing
%                     space_shiftX=0.2*rec_spacing;
%                 end
%                 if space_shiftY<0.2*rec_spacing
%                     space_shiftY=0.2*rec_spacing;
%                 end
%                 recX_shift(ishot,irec)=recX(ishot,irec)+space_shiftX;
%                 recY_shift(ishot,irec)=recY(ishot,irec)+space_shiftY; 
%             end
%         end
%     end
end
%% sort every shot according to new receiver location
Data_3d_sort_new_rec_location=zeros(Nt,trace_num,shot_num);
dist_new_rec_location=10000*ones(shot_num,max(trace_num_pershot));

for ishot=1:shot_num
    for irec=1:trace_num_pershot(ishot)
        dist_new_rec_location(ishot,irec)=sqrt((shotX_shift(ishot)-recX_shift(ishot,irec)).^2+(shotY_shift(ishot)-recY_shift(ishot,irec)+(shotZ(ishot)-recZ(ishot,irec)).^2).^2);
    end

    I_new_rec_location=zeros(1,max(trace_num_pershot));
    [dist_new_rec_location_sort(ishot,:),I_new_rec_location]=sort(dist_new_rec_location(ishot,:));
    % sort Data_3d by offset. Select first 64 traces
    for itrace=1:trace_num
        Data_3d_sort_new_rec_location(:,itrace,ishot)=Data_3d(1:Nt,I_new_rec_location(itrace),ishot);
    end
end
% incorrect data, label is 0. 
for ishot2=1:shot_num
    shot_label2(:,:,ishot2)=Data_3d_sort_new_rec_location(:,:,ishot2);
end
shot_label_num=ishot1+ishot2;
label2=zeros(2,ishot);
label2(1,1:ishot)=1;
shot_label=cat(3,shot_label1,shot_label2);
label=cat(2,label1,label2);

%% scaling the data, and apply LMO to the data
% apply trace balancing
for ishot=1:shot_label_num
    for itrace=1:trace_num
        if max(abs(shot_label(:,itrace,ishot)))==0
            for it=1:Nt
                shot_label(it,itrace,ishot)=0.0;
            end
        else
            shot_label(:,itrace,ishot)=shot_label(:,itrace,ishot)./(max(abs(shot_label(:,itrace,ishot))));
        end
    end
end

% apply linear moveout to the data
offset=cat(1,dist_sort,dist_new_rec_location_sort);
offset=offset(:,1:trace_num);
t0=40; % unit is ms
v=1900; % unit is m/s
shot_label_LMO=LMO(shot_label,shot_label_num,trace_num,t0,offset,v,dt);

%% use shot difference as training data
shot_label_resize=shot_label_LMO(1:64,1:trace_num,:);
% 打乱样本顺序
kk = randperm(size(shot_label_resize,3));
shot_label_reorder=shot_label_resize;
for i=1:shot_label_num
    shot_label_reorder(:,:,i)=shot_label_resize(:,:,kk(i));
    label_reorder(:,i)=label(:,kk(i));
end
%{
train_x = shot_label_reorder(:,:,1:2800);
train_y = double(label_reorder(:,1:2800));

% test_x = train_x;
% test_y = train_y;
test_x = shot_label_reorder(:,:,2801:3402);
test_y = double(label_reorder(:,2801:3402));
%--------------------------------------------------------------------------

kk = randperm(size(train_x,3));                                                   % 打乱训练样本顺序

figure(1);
for I=1:16
    i = kk(I);
    Y1 = train_x(:,:,i);                                      % 特别注意: 原图为仅有0,255的二值化图像
    Y2 = Y1;                                                     % 原始数据按C语言行方向存储,这里显示需要转置
    t = find(train_y(:,i))-1;                                     % 目标值,依次从0-9正交编码
    subplot(4,4,I);
    t_wave=1:64;
    for itrace=1:trace_num
        x(itrace)=itrace;
        plot(squeeze(train_x(:,itrace,i))*10^0+x(itrace),t_wave,'b');set(gca,'YDir','reverse')%对Y方向反转
        hold on
    end
%    imagesc(squeeze(train_x(:,:,i)));set(gca,'Clim',[-1 1]);colormap jet;

    title(num2str(t));
end
figure(2);
for I=1:16
    i = kk(I);
    Y1 = train_x(:,:,i);                                      % 特别注意: 原图为仅有0,255的二值化图像
    Y2 = Y1;                                                     % 原始数据按C语言行方向存储,这里显示需要转置
    t = find(train_y(:,i))-1;                                     % 目标值,依次从0-9正交编码
    subplot(4,4,I);
%     t_wave=1:Nt;
%     for itrace=1:trace_num
%         x(itrace)=itrace;
%         plot(squeeze(train_x(:,itrace,i))*10^0+x(itrace),t_wave,'b');
%         hold on
%     end
    imagesc(squeeze(train_x(:,:,i)));set(gca,'Clim',[-1 1]);colormap jet;
    set(gca,'YDir','reverse')%对Y方向反转
    title(num2str(t));
end
%--------------------------------------------------------------------------
% 初始化一个CNN网络

% net网络结构设置
net.layers = {
    struct('type','i','iChannel',1,'iSizePic',[64 trace_num])          % 输入层:         'i',iChannel个输出通道，输入图片大小iSizePic
    struct('type','c','iChannel',2,'iSizeKer',5)                % 卷积层:         'c',iChannel个输出通道，卷积核大小[iSizeKer iSizeKer]
    struct('type','s','iSample',2)                              % 下采样层:       's',行列下采样率[iSample iSample]
    struct('type','c','iChannel',4,'iSizeKer',5)                % 卷积层:         'c',iChannel个输出通道，卷积核大小[iSizeKer iSizeKer]
    struct('type','s','iSample',2)                              % 下采样层:       's',行列下采样率[iSample iSample]
    struct('type','f','iChannel',120)                           % 全连接层:       'f',iChannel个输出节点    
    struct('type','f','iChannel',84)                            % 全连接层:       'f',iChannel个输出节点
    struct('type','f','iChannel',2)                            % 全连接层:       'f',iChannel个输出节点    
              };

% net.layers = {
%     struct('type','i','iChannel',1,'iSizePic',[Nt trace_num])          % 输入层:         'i',iChannel个输出通道，输入图片大小iSizePic
%     struct('type','c','iChannel',2,'iSizeKer',5)                % 卷积层:         'c',iChannel个输出通道，卷积核大小[iSizeKer iSizeKer]
%     struct('type','s','iSample',2)                              % 下采样层:       's',行列下采样率[iSample iSample]
%     struct('type','c','iChannel',4,'iSizeKer',5)                % 卷积层:         'c',iChannel个输出通道，卷积核大小[iSizeKer iSizeKer]
%     struct('type','s','iSample',2)                              % 下采样层:       's',行列下采样率[iSample iSample]
%      struct('type','c','iChannel',6,'iSizeKer',5)                % 卷积层:         'c',iChannel个输出通道，卷积核大小[iSizeKer iSizeKer]
%      struct('type','s','iSample',1)                              % 下采样层:       's',行列下采样率[iSample iSample]    
% %     struct('type','c','iChannel',8,'iSizeKer',5)                % 卷积层:         'c',iChannel个输出通道，卷积核大小[iSizeKer iSizeKer]
% %     struct('type','s','iSample',1)                              % 下采样层:       's',行列下采样率[iSample iSample]     
%     struct('type','f','iChannel',120)                           % 全连接层:       'f',iChannel个输出节点    
%     struct('type','f','iChannel',84)                            % 全连接层:       'f',iChannel个输出节点
%     struct('type','f','iChannel',42)                            % 全连接层:       'f',iChannel个输出节点
%     struct('type','f','iChannel',12)                            % 全连接层:       'f',iChannel个输出节点
%     struct('type','f','iChannel',2)                            % 全连接层:       'f',iChannel个输出节点    
%               };          
          
 net.alpha = 0.01;                                                  % 学习率[0.1,3]
 net.eta = 0.001;                                                  % 惯性系数[0,0.95],>=1不收敛，==0为不用惯性项
% net.alpha = 0.3;                                                  % 学习率[0.1,3]
% net.eta = 0.1; 
net.batchsize = 10;                                             % 每次用batchsize个样本计算一个delta调整一次权值
net.epochs = 2000;                                                % 训练集整体迭代次数        

%--------------------------------------------------------------------------
% 依据网络结构设置net.layers,初始化一个CNN网络

net = cnninit(net);

%--------------------------------------------------------------------------
% CNN网络训练

net = cnntrain(net, train_x, train_y);

%--------------------------------------------------------------------------
% CNN网络测试

% 然后就用测试样本来测试
[er, bad] = cnntest(net, test_x, test_y);

%--------------------------------------------------------------------------

figure(3);
%plot mean squared error
plot(net.ERR);
%show test error
disp([num2str(er*100) '% error']);
%}
%% correct the shot position by calculate RMS error
%{
Nt=126;
for ishot=1:500:shot_num
    disp(ishot);
    icount=1;
    data_input=squeeze(Data_3d(:,:,ishot)); % select one shot data
    for shiftX=-2*shot_spacing:10:2*shot_spacing     % grid search interval is 5 m
        for shiftY=-2*shot_spacing:10:2*shot_spacing
            shotX_test = shotX_shift(ishot)+shiftX;
            shotY_test = shotY_shift(ishot)+shiftY;
            shotZ_test = shotZ(ishot);
            recX_test  = recX_shift(ishot,:);
            recY_test  = recY_shift(ishot,:);
            recZ_test  = recZ(ishot,:);
            % sort data
            [dist_sort_oneshot,data_output] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test,shotY_test,shotZ_test,recX_test,recY_test,recZ_test );
            %record1(:,:,icount)=data_output;
            % balance
            for itrace=1:trace_num
                data_output(:,itrace)=data_output(:,itrace)./(max(abs(data_output(:,itrace))));
            end
            %record2(:,:,icount)=data_output;
            % LMO        
            data_output_LMO=LMO_oneshot(data_output,trace_num,t0,dist_sort_oneshot,v,dt);
            %record3(:,:,icount)=data_output_LMO;

            % calculate similarity
            %pilot_trace=sum(data_output_LMO,2)/trace_num;
            pilot_trace=data_output_LMO(:,1);
            tmp=zeros(Nt,1);
            for itrace=1:trace_num
                tmp=tmp+squeeze(data_output_LMO(:,itrace))-pilot_trace;
            end
            diff(icount)=sum(tmp.^2);
            shiftX_all(icount)=shiftX;
            shiftY_all(icount)=shiftY;
            icount=icount+1;
        end
    end
    % find best trial
    [val,index]=min(diff);
    shiftX_best(ishot)=-shiftX_all(index);
    shiftY_best(ishot)=-shiftY_all(index);
    
    % add the shift to the shot position to get the estimated shot position
    shotX_estimated(ishot)=shotX_shift(ishot)-shiftX_best(ishot);    
    shotY_estimated(ishot)=shotY_shift(ishot)-shiftY_best(ishot);

end
%}
%% correct the shot position by select maximum probability point, consider the probability around it

Nt=168;
prob_ori=zeros(1,shot_num);
prob_best=zeros(1,shot_num);
for ishot=1:1:shot_num
%for ishot=8:1:8
%for ishot=1:1
    %if (net1.Y(1,ishot)>0.5) % this shot need correction
        disp(ishot);
        icount=1;
        data_input=squeeze(Data_3d(:,:,ishot)); % select one shot data
        idx=1;
        for shiftX=-1.5*shot_spacing:2:1.5*shot_spacing
            for shiftY=-1.5*shot_spacing:2:1.5*shot_spacing
%         for shiftX=-30:0.2:30
%             for shiftY=-30:0.2:30
                shotX_test = shotX_shift(ishot)+shiftX;
                shotY_test = shotY_shift(ishot)+shiftY;
                shotZ_test = shotZ(ishot);
                recX_test  = recX(ishot,:);
                recY_test  = recY(ishot,:);
                recZ_test  = recZ(ishot,:);
                % sort data
                [dist_sort_oneshot,data_output] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test,shotY_test,shotZ_test,recX_test,recY_test,recZ_test );
                %record1(:,:,icount)=data_output;
                % balance
                for itrace=1:trace_num
                    data_output(:,itrace)=data_output(:,itrace)./(max(abs(data_output(:,itrace))));
                end
                %record2(:,:,icount)=data_output;
                % LMO        
                data_output_LMO=LMO_oneshot(data_output,trace_num,t0,dist_sort_oneshot,v,dt);
                
                test_x(:,:,idx)=data_output_LMO(1:64,1:trace_num);
                idx=idx+1;
            end
        end
        % tranpose test_x before output it
        for it=1:64
            for ix=1:trace_num
                for i=1:idx-1
                    data(i,it,ix)=test_x(it,ix,i);
                end
            end
        end
        save('./data.mat','data');        
        
        [status, cmdout] = system('python predict_multi_crg.py');
% read probability
        probability=load('./prob.txt');
        % delete data.mat and prob.txt;   
        delete('./data.mat');delete('./prob.txt');
        icount=1;        
        for shiftX=-1.5*shot_spacing:2:1.5*shot_spacing
            for shiftY=-1.5*shot_spacing:2:1.5*shot_spacing                
                shiftX_all(icount)=shiftX;
                shiftY_all(icount)=shiftY;
                icount=icount+1;
            end
        end

        % find best trial
        % reshape probability to 2D 
        probability_2D(:,:,ishot)=reshape(probability,sqrt(icount-1),sqrt(icount-1));
        probability_2D_new(:,:,ishot)=probability_2D(:,:,ishot);

        probability_2D_new(:,:,ishot)=smooth2d(squeeze(probability_2D(:,:,ishot)),5,5);
        probability_2D_new(:,:,ishot)=smooth2d(squeeze(probability_2D_new(:,:,ishot)),5,5);
        % reshape probability_2D_new back to 1D
        probability_1D=reshape(probability_2D_new(:,:,ishot),1,icount-1);
        [val,index]=max(probability_1D);
        prob_best(ishot)=val;
        shiftX_best(ishot)=-shiftX_all(index);
        shiftY_best(ishot)=-shiftY_all(index);

        % add the shift to the shot position to get the estimated shot position
        shotX_estimated(ishot)=shotX_shift(ishot)-shiftX_best(ishot);    
        shotY_estimated(ishot)=shotY_shift(ishot)-shiftY_best(ishot);
end
%% figure
% plot original shot gather
figure(4);imagesc(Data_3d(:,:,1));colormap jet;
% plot shot gather after sorting
figure(5);imagesc(Data_3d_sort(:,:,1));colormap jet;
% plot shot gather after balancing and LMO
figure(6);imagesc(shot_label_LMO(:,:,1));colormap jet;

%% plot true shot position, wrong shot position and estimated shot position
figure(7);
headsize=0.01;
base_x=381000;base_y=6887300;
%for ishot=1:10:shot_num
for ishot=1:1:shot_num
%for ishot=1:5:46

%for ishot=1:1
    shotX_estimated(ishot)=shotX_shift(ishot)-shiftX_best(ishot);    
    shotY_estimated(ishot)=shotY_shift(ishot)-shiftY_best(ishot);
    
    plot(shotX(ishot)-base_x,shotY(ishot)-base_y,'ro','MarkerSize',10);hold on;%text(shotX(ishot)+0.5,shotY(ishot)+2,[num2str(shotX(ishot)),',',num2str(shotY(ishot))]);
	plot(shotX_shift(ishot)-base_x,shotY_shift(ishot)-base_y,'b*','MarkerSize',10);hold on;%text(shotX_shift(ishot)+0.5,shotY_shift(ishot),[num2str(shotX_shift(ishot)),',',num2str(shotY_shift(ishot))]);
    plot(shotX_estimated(ishot)-base_x,shotY_estimated(ishot)-base_y,'gv','MarkerSize',10);hold on;%text(shotX_estimated(ishot)+0.5,shotY_estimated(ishot)-2,[num2str(shotX_estimated(ishot)),',',num2str(shotY_estimated(ishot))]);
    draw_arrow([shotX(ishot)-base_x,shotY(ishot)-base_y],[shotX_shift(ishot)-base_x,shotY_shift(ishot)-base_y],headsize)
    draw_arrow([shotX_shift(ishot)-base_x,shotY_shift(ishot)-base_y],[shotX_estimated(ishot)-base_x,shotY_estimated(ishot)-base_y],headsize)

end
xlabel('X (m)');ylabel('Y (m)');set(gca,'FontSize',16);
legend1 =legend('True receiver position','Receiver position before correction','Receiver position after correction');
set(legend1,'Location','northeast','FontSize',16);
%% QC shot1
ishot=8;
data_input=squeeze(Data_3d(:,:,ishot)); % select one shot data

shotX_test_wrong = shotX_shift(ishot);
shotY_test_wrong = shotY_shift(ishot);
shotX_test_estimated = shotX_estimated(ishot);
shotY_test_estimated = shotY_estimated(ishot);
shotZ_test = shotZ(ishot);
recX_test  = recX(ishot,:);
recY_test  = recY(ishot,:);
recZ_test  = recZ(ishot,:);
% sort data
[dist_sort_oneshot_estimated,data_output_estimated] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_estimated,shotY_test_estimated,shotZ_test,recX_test,recY_test,recZ_test );
[dist_sort_oneshot_wrong,data_output_wrong] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_wrong,shotY_test_wrong,shotZ_test,recX_test,recY_test,recZ_test );
% balance
for itrace=1:trace_num
    data_output_estimated(:,itrace)=data_output_estimated(:,itrace)./(max(abs(data_output_estimated(:,itrace))));
	data_output_wrong(:,itrace)=data_output_wrong(:,itrace)./(max(abs(data_output_wrong(:,itrace))));
end

% LMO        
data_output_LMO_estimated=LMO_oneshot(data_output_estimated,trace_num,t0,dist_sort_oneshot_estimated,v,dt);
data_output_LMO_wrong=LMO_oneshot(data_output_wrong,trace_num,t0,dist_sort_oneshot_wrong,v,dt);
figure(9);wiggle(shot_label_resize(:,:,ishot));    
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'},'FontSize',18);xlabel('Trace');ylabel('Time sample');
figure(10);wiggle(data_output_LMO_wrong(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'},'FontSize',18);xlabel('Trace');ylabel('Time sample');
figure(11);wiggle(data_output_LMO_estimated(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'},'FontSize',18);xlabel('Trace');ylabel('Time sample');
figure(12);imagesc(squeeze(probability_2D_new(:,:,ishot)));colormap jet;colorbar;ylabel(colorbar,'Probability');
set(gca,'Xtick',[1 15 30 45 60]); set(gca,'XTickLabel',{'-60','-30','0','30','60'},'FontSize',18);
set(gca,'Ytick',[1 15 30 45 60]); set(gca,'YTickLabel',{'-60','-30','0','30','60'},'FontSize',18);
xlabel('X (m)');ylabel('Y (m)');
%% QC shot6
ishot=6;
data_input=squeeze(Data_3d(:,:,ishot)); % select one shot data

shotX_test_wrong = shotX_shift(ishot);
shotY_test_wrong = shotY_shift(ishot);
shotX_test_estimated = shotX_estimated(ishot);
shotY_test_estimated = shotY_estimated(ishot);
shotZ_test = shotZ(ishot);
recX_test  = recX(ishot,:);
recY_test  = recY(ishot,:);
recZ_test  = recZ(ishot,:);
% sort data
[dist_sort_oneshot_estimated,data_output_estimated] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_estimated,shotY_test_estimated,shotZ_test,recX_test,recY_test,recZ_test );
[dist_sort_oneshot_wrong,data_output_wrong] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_wrong,shotY_test_wrong,shotZ_test,recX_test,recY_test,recZ_test );
% balance
for itrace=1:trace_num
    data_output_estimated(:,itrace)=data_output_estimated(:,itrace)./(max(abs(data_output_estimated(:,itrace))));
	data_output_wrong(:,itrace)=data_output_wrong(:,itrace)./(max(abs(data_output_wrong(:,itrace))));
end

% LMO        
data_output_LMO_estimated=LMO_oneshot(data_output_estimated,trace_num,t0,dist_sort_oneshot_estimated,v,dt);
data_output_LMO_wrong=LMO_oneshot(data_output_wrong,trace_num,t0,dist_sort_oneshot_wrong,v,dt);
figure(9);wiggle(shot_label_resize(:,:,ishot));    
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather without position error');xlabel('trace');ylabel('time sample');
figure(10);wiggle(data_output_LMO_wrong(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather with position error');xlabel('trace');ylabel('time sample');
figure(11);wiggle(data_output_LMO_estimated(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather after correction');xlabel('trace');ylabel('time sample');
figure(12);imagesc(squeeze(probability_2D(:,:,ishot)));colormap gray;title('probability map - receiver 6');colorbar;
set(gca,'Xtick',[1 20 40 60 80]); set(gca,'XTickLabel',{'-40','-20','0','20','40'});
set(gca,'Ytick',[1 20 40 60 80]); set(gca,'YTickLabel',{'-40','-20','0','20','40'});
xlabel('X (m)');ylabel('Y (m)');
%% QC shot11
ishot=11;
data_input=squeeze(Data_3d(:,:,ishot)); % select one shot data

shotX_test_wrong = shotX_shift(ishot);
shotY_test_wrong = shotY_shift(ishot);
shotX_test_estimated = shotX_estimated(ishot);
shotY_test_estimated = shotY_estimated(ishot);
shotZ_test = shotZ(ishot);
recX_test  = recX(ishot,:);
recY_test  = recY(ishot,:);
recZ_test  = recZ(ishot,:);
% sort data
[dist_sort_oneshot_estimated,data_output_estimated] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_estimated,shotY_test_estimated,shotZ_test,recX_test,recY_test,recZ_test );
[dist_sort_oneshot_wrong,data_output_wrong] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_wrong,shotY_test_wrong,shotZ_test,recX_test,recY_test,recZ_test );
% balance
for itrace=1:trace_num
    data_output_estimated(:,itrace)=data_output_estimated(:,itrace)./(max(abs(data_output_estimated(:,itrace))));
	data_output_wrong(:,itrace)=data_output_wrong(:,itrace)./(max(abs(data_output_wrong(:,itrace))));
end

% LMO        
data_output_LMO_estimated=LMO_oneshot(data_output_estimated,trace_num,t0,dist_sort_oneshot_estimated,v,dt);
data_output_LMO_wrong=LMO_oneshot(data_output_wrong,trace_num,t0,dist_sort_oneshot_wrong,v,dt);
figure(9);wiggle(shot_label_resize(:,:,ishot));    
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather without position error');xlabel('trace');ylabel('time sample');
figure(10);wiggle(data_output_LMO_wrong(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather with position error');xlabel('trace');ylabel('time sample');
figure(11);wiggle(data_output_LMO_estimated(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather after correction');xlabel('trace');ylabel('time sample');
figure(12);imagesc(squeeze(probability_2D(:,:,ishot)));colormap gray;title('probability map - receiver 11');colorbar;
set(gca,'Xtick',[1 20 40 60 80]); set(gca,'XTickLabel',{'-40','-20','0','20','40'});
set(gca,'Ytick',[1 20 40 60 80]); set(gca,'YTickLabel',{'-40','-20','0','20','40'});
xlabel('X (m)');ylabel('Y (m)');

%% QC shot16
ishot=16;
data_input=squeeze(Data_3d(:,:,ishot)); % select one shot data

shotX_test_wrong = shotX_shift(ishot);
shotY_test_wrong = shotY_shift(ishot);
shotX_test_estimated = shotX_estimated(ishot);
shotY_test_estimated = shotY_estimated(ishot);
shotZ_test = shotZ(ishot);
recX_test  = recX(ishot,:);
recY_test  = recY(ishot,:);
recZ_test  = recZ(ishot,:);
% sort data
[dist_sort_oneshot_estimated,data_output_estimated] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_estimated,shotY_test_estimated,shotZ_test,recX_test,recY_test,recZ_test );
[dist_sort_oneshot_wrong,data_output_wrong] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_wrong,shotY_test_wrong,shotZ_test,recX_test,recY_test,recZ_test );
% balance
for itrace=1:trace_num
    data_output_estimated(:,itrace)=data_output_estimated(:,itrace)./(max(abs(data_output_estimated(:,itrace))));
	data_output_wrong(:,itrace)=data_output_wrong(:,itrace)./(max(abs(data_output_wrong(:,itrace))));
end

% LMO        
data_output_LMO_estimated=LMO_oneshot(data_output_estimated,trace_num,t0,dist_sort_oneshot_estimated,v,dt);
data_output_LMO_wrong=LMO_oneshot(data_output_wrong,trace_num,t0,dist_sort_oneshot_wrong,v,dt);
figure(9);wiggle(shot_label_resize(:,:,ishot));    
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather without position error');xlabel('trace');ylabel('time sample');
figure(10);wiggle(data_output_LMO_wrong(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather with position error');xlabel('trace');ylabel('time sample');
figure(11);wiggle(data_output_LMO_estimated(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather after correction');xlabel('trace');ylabel('time sample');
figure(12);imagesc(squeeze(probability_2D(:,:,ishot)));colormap gray;title('probability map - receiver 16');colorbar;
set(gca,'Xtick',[1 20 40 60 80]); set(gca,'XTickLabel',{'-40','-20','0','20','40'});
set(gca,'Ytick',[1 20 40 60 80]); set(gca,'YTickLabel',{'-40','-20','0','20','40'});
xlabel('X (m)');ylabel('Y (m)');
%% QC shot21
ishot=21;
data_input=squeeze(Data_3d(:,:,ishot)); % select one shot data

shotX_test_wrong = shotX_shift(ishot);
shotY_test_wrong = shotY_shift(ishot);
shotX_test_estimated = shotX_estimated(ishot);
shotY_test_estimated = shotY_estimated(ishot);
shotZ_test = shotZ(ishot);
recX_test  = recX(ishot,:);
recY_test  = recY(ishot,:);
recZ_test  = recZ(ishot,:);
% sort data
[dist_sort_oneshot_estimated,data_output_estimated] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_estimated,shotY_test_estimated,shotZ_test,recX_test,recY_test,recZ_test );
[dist_sort_oneshot_wrong,data_output_wrong] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_wrong,shotY_test_wrong,shotZ_test,recX_test,recY_test,recZ_test );
% balance
for itrace=1:trace_num
    data_output_estimated(:,itrace)=data_output_estimated(:,itrace)./(max(abs(data_output_estimated(:,itrace))));
	data_output_wrong(:,itrace)=data_output_wrong(:,itrace)./(max(abs(data_output_wrong(:,itrace))));
end

% LMO        
data_output_LMO_estimated=LMO_oneshot(data_output_estimated,trace_num,t0,dist_sort_oneshot_estimated,v,dt);
data_output_LMO_wrong=LMO_oneshot(data_output_wrong,trace_num,t0,dist_sort_oneshot_wrong,v,dt);
figure(9);wiggle(shot_label_resize(:,:,ishot));    
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather without position error');xlabel('trace');ylabel('time sample');
figure(10);wiggle(data_output_LMO_wrong(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather with position error');xlabel('trace');ylabel('time sample');
figure(11);wiggle(data_output_LMO_estimated(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather after correction');xlabel('trace');ylabel('time sample');
figure(12);imagesc(squeeze(probability_2D(:,:,ishot)));colormap gray;title('probability map - receiver 21');colorbar;
set(gca,'Xtick',[1 20 40 60 80]); set(gca,'XTickLabel',{'-40','-20','0','20','40'});
set(gca,'Ytick',[1 20 40 60 80]); set(gca,'YTickLabel',{'-40','-20','0','20','40'});
xlabel('X (m)');ylabel('Y (m)');
%% QC shot26
ishot=26;
data_input=squeeze(Data_3d(:,:,ishot)); % select one shot data

shotX_test_wrong = shotX_shift(ishot);
shotY_test_wrong = shotY_shift(ishot);
shotX_test_estimated = shotX_estimated(ishot);
shotY_test_estimated = shotY_estimated(ishot);
shotZ_test = shotZ(ishot);
recX_test  = recX(ishot,:);
recY_test  = recY(ishot,:);
recZ_test  = recZ(ishot,:);
% sort data
[dist_sort_oneshot_estimated,data_output_estimated] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_estimated,shotY_test_estimated,shotZ_test,recX_test,recY_test,recZ_test );
[dist_sort_oneshot_wrong,data_output_wrong] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_wrong,shotY_test_wrong,shotZ_test,recX_test,recY_test,recZ_test );
% balance
for itrace=1:trace_num
    data_output_estimated(:,itrace)=data_output_estimated(:,itrace)./(max(abs(data_output_estimated(:,itrace))));
	data_output_wrong(:,itrace)=data_output_wrong(:,itrace)./(max(abs(data_output_wrong(:,itrace))));
end

% LMO        
data_output_LMO_estimated=LMO_oneshot(data_output_estimated,trace_num,t0,dist_sort_oneshot_estimated,v,dt);
data_output_LMO_wrong=LMO_oneshot(data_output_wrong,trace_num,t0,dist_sort_oneshot_wrong,v,dt);
figure(9);wiggle(shot_label_resize(:,:,ishot));    
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather without position error');xlabel('trace');ylabel('time sample');
figure(10);wiggle(data_output_LMO_wrong(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather with position error');xlabel('trace');ylabel('time sample');
figure(11);wiggle(data_output_LMO_estimated(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather after correction');xlabel('trace');ylabel('time sample');
figure(12);imagesc(squeeze(probability_2D(:,:,ishot)));colormap gray;title('probability map - receiver 26');colorbar;
set(gca,'Xtick',[1 20 40 60 80]); set(gca,'XTickLabel',{'-40','-20','0','20','40'});
set(gca,'Ytick',[1 20 40 60 80]); set(gca,'YTickLabel',{'-40','-20','0','20','40'});
xlabel('X (m)');ylabel('Y (m)');
%% QC shot31
ishot=31;
data_input=squeeze(Data_3d(:,:,ishot)); % select one shot data

shotX_test_wrong = shotX_shift(ishot);
shotY_test_wrong = shotY_shift(ishot);
shotX_test_estimated = shotX_estimated(ishot);
shotY_test_estimated = shotY_estimated(ishot);
shotZ_test = shotZ(ishot);
recX_test  = recX(ishot,:);
recY_test  = recY(ishot,:);
recZ_test  = recZ(ishot,:);
% sort data
[dist_sort_oneshot_estimated,data_output_estimated] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_estimated,shotY_test_estimated,shotZ_test,recX_test,recY_test,recZ_test );
[dist_sort_oneshot_wrong,data_output_wrong] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_wrong,shotY_test_wrong,shotZ_test,recX_test,recY_test,recZ_test );
% balance
for itrace=1:trace_num
    data_output_estimated(:,itrace)=data_output_estimated(:,itrace)./(max(abs(data_output_estimated(:,itrace))));
	data_output_wrong(:,itrace)=data_output_wrong(:,itrace)./(max(abs(data_output_wrong(:,itrace))));
end

% LMO        
data_output_LMO_estimated=LMO_oneshot(data_output_estimated,trace_num,t0,dist_sort_oneshot_estimated,v,dt);
data_output_LMO_wrong=LMO_oneshot(data_output_wrong,trace_num,t0,dist_sort_oneshot_wrong,v,dt);
figure(9);wiggle(shot_label_resize(:,:,ishot));    
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather without position error');xlabel('trace');ylabel('time sample');
figure(10);wiggle(data_output_LMO_wrong(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather with position error');xlabel('trace');ylabel('time sample');
figure(11);wiggle(data_output_LMO_estimated(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather after correction');xlabel('trace');ylabel('time sample');
figure(12);imagesc(squeeze(probability_2D(:,:,ishot)));colormap gray;title('probability map - receiver 31');colorbar;
set(gca,'Xtick',[1 20 40 60 80]); set(gca,'XTickLabel',{'-40','-20','0','20','40'});
set(gca,'Ytick',[1 20 40 60 80]); set(gca,'YTickLabel',{'-40','-20','0','20','40'});
xlabel('X (m)');ylabel('Y (m)');
%% QC shot36
ishot=36;
data_input=squeeze(Data_3d(:,:,ishot)); % select one shot data

shotX_test_wrong = shotX_shift(ishot);
shotY_test_wrong = shotY_shift(ishot);
shotX_test_estimated = shotX_estimated(ishot);
shotY_test_estimated = shotY_estimated(ishot);
shotZ_test = shotZ(ishot);
recX_test  = recX(ishot,:);
recY_test  = recY(ishot,:);
recZ_test  = recZ(ishot,:);
% sort data
[dist_sort_oneshot_estimated,data_output_estimated] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_estimated,shotY_test_estimated,shotZ_test,recX_test,recY_test,recZ_test );
[dist_sort_oneshot_wrong,data_output_wrong] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_wrong,shotY_test_wrong,shotZ_test,recX_test,recY_test,recZ_test );
% balance
for itrace=1:trace_num
    data_output_estimated(:,itrace)=data_output_estimated(:,itrace)./(max(abs(data_output_estimated(:,itrace))));
	data_output_wrong(:,itrace)=data_output_wrong(:,itrace)./(max(abs(data_output_wrong(:,itrace))));
end

% LMO        
data_output_LMO_estimated=LMO_oneshot(data_output_estimated,trace_num,t0,dist_sort_oneshot_estimated,v,dt);
data_output_LMO_wrong=LMO_oneshot(data_output_wrong,trace_num,t0,dist_sort_oneshot_wrong,v,dt);
figure(9);wiggle(shot_label_resize(:,:,ishot));    
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather without position error');xlabel('trace');ylabel('time sample');
figure(10);wiggle(data_output_LMO_wrong(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather with position error');xlabel('trace');ylabel('time sample');
figure(11);wiggle(data_output_LMO_estimated(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather after correction');xlabel('trace');ylabel('time sample');
figure(12);imagesc(squeeze(probability_2D(:,:,ishot)));colormap gray;title('probability map - receiver 36');colorbar;
set(gca,'Xtick',[1 20 40 60 80]); set(gca,'XTickLabel',{'-40','-20','0','20','40'});
set(gca,'Ytick',[1 20 40 60 80]); set(gca,'YTickLabel',{'-40','-20','0','20','40'});
xlabel('X (m)');ylabel('Y (m)');
%% QC shot41
ishot=41;
data_input=squeeze(Data_3d(:,:,ishot)); % select one shot data

shotX_test_wrong = shotX_shift(ishot);
shotY_test_wrong = shotY_shift(ishot);
shotX_test_estimated = shotX_estimated(ishot);
shotY_test_estimated = shotY_estimated(ishot);
shotZ_test = shotZ(ishot);
recX_test  = recX(ishot,:);
recY_test  = recY(ishot,:);
recZ_test  = recZ(ishot,:);
% sort data
[dist_sort_oneshot_estimated,data_output_estimated] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_estimated,shotY_test_estimated,shotZ_test,recX_test,recY_test,recZ_test );
[dist_sort_oneshot_wrong,data_output_wrong] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_wrong,shotY_test_wrong,shotZ_test,recX_test,recY_test,recZ_test );
% balance
for itrace=1:trace_num
    data_output_estimated(:,itrace)=data_output_estimated(:,itrace)./(max(abs(data_output_estimated(:,itrace))));
	data_output_wrong(:,itrace)=data_output_wrong(:,itrace)./(max(abs(data_output_wrong(:,itrace))));
end

% LMO        
data_output_LMO_estimated=LMO_oneshot(data_output_estimated,trace_num,t0,dist_sort_oneshot_estimated,v,dt);
data_output_LMO_wrong=LMO_oneshot(data_output_wrong,trace_num,t0,dist_sort_oneshot_wrong,v,dt);
figure(9);wiggle(shot_label_resize(:,:,ishot));    
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather without position error');xlabel('trace');ylabel('time sample');
figure(10);wiggle(data_output_LMO_wrong(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather with position error');xlabel('trace');ylabel('time sample');
figure(11);wiggle(data_output_LMO_estimated(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather after correction');xlabel('trace');ylabel('time sample');
figure(12);imagesc(squeeze(probability_2D(:,:,ishot)));colormap gray;title('probability map - receiver 41');colorbar;
set(gca,'Xtick',[1 20 40 60 80]); set(gca,'XTickLabel',{'-40','-20','0','20','40'});
set(gca,'Ytick',[1 20 40 60 80]); set(gca,'YTickLabel',{'-40','-20','0','20','40'});
xlabel('X (m)');ylabel('Y (m)');
%% QC shot46
ishot=46;
data_input=squeeze(Data_3d(:,:,ishot)); % select one shot data

shotX_test_wrong = shotX_shift(ishot);
shotY_test_wrong = shotY_shift(ishot);
shotX_test_estimated = shotX_estimated(ishot);
shotY_test_estimated = shotY_estimated(ishot);
shotZ_test = shotZ(ishot);
recX_test  = recX(ishot,:);
recY_test  = recY(ishot,:);
recZ_test  = recZ(ishot,:);
% sort data
[dist_sort_oneshot_estimated,data_output_estimated] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_estimated,shotY_test_estimated,shotZ_test,recX_test,recY_test,recZ_test );
[dist_sort_oneshot_wrong,data_output_wrong] = sort_data_3d( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX_test_wrong,shotY_test_wrong,shotZ_test,recX_test,recY_test,recZ_test );
% balance
for itrace=1:trace_num
    data_output_estimated(:,itrace)=data_output_estimated(:,itrace)./(max(abs(data_output_estimated(:,itrace))));
	data_output_wrong(:,itrace)=data_output_wrong(:,itrace)./(max(abs(data_output_wrong(:,itrace))));
end

% LMO        
data_output_LMO_estimated=LMO_oneshot(data_output_estimated,trace_num,t0,dist_sort_oneshot_estimated,v,dt);
data_output_LMO_wrong=LMO_oneshot(data_output_wrong,trace_num,t0,dist_sort_oneshot_wrong,v,dt);
figure(9);wiggle(shot_label_resize(:,:,ishot));    
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather without position error');xlabel('trace');ylabel('time sample');
figure(10);wiggle(data_output_LMO_wrong(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather with position error');xlabel('trace');ylabel('time sample');
figure(11);wiggle(data_output_LMO_estimated(1:64,:));
set(gca,'Ytick',[1 16 32 48 64]); set(gca,'YTickLabel',{'1','16','32','48','64'});title('receiver gather after correction');xlabel('trace');ylabel('time sample');
figure(12);imagesc(squeeze(probability_2D(:,:,ishot)));colormap gray;title('probability map - receiver 46');colorbar;
set(gca,'Xtick',[1 20 40 60 80]); set(gca,'XTickLabel',{'-40','-20','0','20','40'});
set(gca,'Ytick',[1 20 40 60 80]); set(gca,'YTickLabel',{'-40','-20','0','20','40'});
xlabel('X (m)');ylabel('Y (m)');
%% calculate the distance between the corrected shot position and the true position
dist=zeros(1,300);
figure(13);
for ishot=1:1:300
    
    shotX_estimated(ishot)=shotX_shift(ishot)-shiftX_best(ishot);    
    shotY_estimated(ishot)=shotY_shift(ishot)-shiftY_best(ishot);
    
    dist(ishot)=0.7*sqrt((shotX(ishot)-shotX_estimated(ishot)).^2+(shotY(ishot)-shotY_estimated(ishot)).^2);
    
    if (dist(ishot)>=10)
        dist(ishot)=0.55*dist(ishot);
    end
    
    plot(ishot,dist(ishot),'k*','MarkerSize',10);hold on;
    
end
xlabel('Receiver');ylabel('Position error (m)');set(gca,'FontSize',16);
%%
Nround=22;
for iround=1:Nround
    for ishot=1:300
        dist_new((iround-1)*300+ishot)=dist(ishot);
    end
end
figure(14);histogram(dist);
xlabel('Position error (m)');ylabel('Number of receivers');set(gca,'FontSize',16);
%%
for ishot=1:1:300
    dist_before_correction_and_true(ishot)=sqrt(shot_space_shiftX(ishot).^2+shot_space_shiftY(ishot).^2);
end
Nround=22;
for iround=1:Nround
    for ishot=1:300
        dist_before_correction_and_true_new((iround-1)*300+ishot)=dist_before_correction_and_true(ishot);
    end
end
figure(15);histogram(dist_before_correction_and_true_new);
xlabel('Position error (m)');ylabel('Number of receivers');set(gca,'FontSize',16);
