function [ dist_sort_oneshot,data_output ] = sort_data( data_input,Nt,trace_num,trace_num_pershot,ishot,shotX,shotY,recX,recY )
% sort seismic data according to the offset, sort only 1 shot
%   data_input: input 2D data, Data_input(it,irec)
%   data_output:outputput 2D data, data_output(it,irec)
% shotX, shotY, recX, recY: position of shot and rec

data_output=zeros(Nt,trace_num);
dist=1000*ones(trace_num_pershot(ishot));

    for irec=1:trace_num_pershot(ishot)
        dist(irec)=sqrt((shotX-recX(irec)).^2+(shotY-recY(irec)).^2);
    end
    
    I=zeros(1,trace_num_pershot(ishot));
    [dist_sort,I]=sort(dist(:));
    % sort Data_3d by offset. Select first 64 traces
    for itrace=1:trace_num
        data_output(:,itrace)=data_input(1:Nt,I(itrace));
        dist_sort_oneshot(itrace)=dist_sort(itrace);
    end


end

