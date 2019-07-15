% calculate shot position error diagram
shot_num=405;
shot_spacing=100;
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
    
    dist(ishot)=sqrt(shot_space_shiftX(ishot).^2+shot_space_shiftY(ishot).^2);
end

figure(1);histogram(dist);
xlabel('Position error (m)');ylabel('Number of shots');set(gca,'FontSize',16);