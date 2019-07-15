function [ shot_LMO ] = LMO_oneshot( shot, trace_num, t0, offset, v, dt )
% apply linear moveout to the shot gather
%   shot is the input shot gather, it is a 2D array. shot_label(it,irec)

%   t_LMO=t0+offset/v

    for irec=1:trace_num
        t_LMO(irec)=-t0+1000.0*offset(irec)/v; % unit is ms.
        t_LMO(irec)=round(t_LMO(irec)/dt);
        shot_LMO(:,irec)=shot(mod((1:end)+t_LMO(irec)-1, end)+1,irec);
    end

end

