function [ shot_LMO ] = LMO( shot, shot_num,trace_num, t0, offset, v, dt )
% apply linear moveout to the shot gather
%   shot is the input shot gather, it is a 3D array. shot_label(it,irec,ishot)
%   shot_num is the number of shot
%   t_LMO=t0+offset/v
for ishot=1:shot_num
    for irec=1:trace_num
        t_LMO(ishot,irec)=-t0+1000.0*offset(ishot,irec)/v; % unit is ms.
        t_LMO(ishot,irec)=round(t_LMO(ishot,irec)/dt);
        shot_LMO(:,irec,ishot)=shot(mod((1:end)+t_LMO(ishot,irec)-1, end)+1,irec,ishot);
    end
end


end

