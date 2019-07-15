%% use elevation information when calculate offset
clear; close all; clc;
Nround=22;
%% read seismic data
shot_num = 300; % shot number
trace_num = 96; % trace number per shot
rec_spacing = 100; % receiver spacing, unit=m
shot_spacing = 40; % shot spacing, unit=m
dt=6; % unit=ms 
[Data_2d,SegyTraceHeaders,SegyHeader] = ReadSegy('../../data/2007_merge_sk_crg_96trace.sgy'); Nt=168;
disp('read data end');
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
for iround=1:Nround
    
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
    shot_space_shiftX(ishot)=2*(rand(1,1)-0.5)*shot_spacing;  % range from -1*shot_spacing to 1*shot_spacing
    shot_space_shiftY(ishot)=2*(rand(1,1)-0.5)*shot_spacing;  % range from -1*shot_spacing to 1*shot_spacing
	while( shot_space_shiftX(ishot)>=-0.2*shot_spacing && shot_space_shiftX(ishot)<=0.2*shot_spacing )
        shot_space_shiftX(ishot)=2*(rand(1,1)-0.5)*shot_spacing;
    end

	while( shot_space_shiftY(ishot)>=-0.2*shot_spacing && shot_space_shiftY(ishot)<=0.2*shot_spacing )
        shot_space_shiftY(ishot)=2*(rand(1,1)-0.5)*shot_spacing;
    end

    
    shotX_shift(ishot)=shotX(ishot)+shot_space_shiftX(ishot);
    shotY_shift(ishot)=shotY(ishot)+shot_space_shiftY(ishot);    
    
    random_number=randperm(max_random_number,1); % create a random number that range from 1 to max_random_number (5).
    rec=randperm(trace_num,random_number); % select random_number receivers to change the location

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
% ´òÂÒÑù±¾Ë³Ðò
kk = randperm(size(shot_label_resize,3));
shot_label_reorder=shot_label_resize;
for i=1:shot_label_num
    shot_label_reorder(:,:,i,iround)=shot_label_resize(:,:,kk(i));
    label_reorder(:,i,iround)=label(:,kk(i));
end

end
%%
tmp1=squeeze(shot_label_reorder(:,:,:,1));
tmp2=squeeze(label_reorder(:,:,1));
for iround=1:Nround-1
    tmp1=cat(3,tmp1,squeeze(shot_label_reorder(:,:,:,iround+1)));
    tmp2=cat(2,tmp2,squeeze(label_reorder(:,:,iround+1)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% convert train_x/y,test_x/y for keras
for it=1:64
    for ix=1:96
        for i=1:13200
            tmp3(i,it,ix)=tmp1(it,ix,i);
        end 
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shot_label_final=tmp3(:,:,:);
label_final=tmp2(1,:)';