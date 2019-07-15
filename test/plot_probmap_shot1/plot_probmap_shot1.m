a=load('matlab_code.mat');
b=load('keras_code.mat');
c=0.7*a.probability_2D_new+0.3*b.probability_2D_new;
figure(12);imagesc(squeeze(c(:,:,ishot)));colormap gray;colorbar;ylabel(colorbar,'Probability');
set(gca,'Xtick',[1 37 75 112 150]); set(gca,'XTickLabel',{'-150','-75','0','75','150'},'FontSize',18);
set(gca,'Ytick',[1 37 75 112 150]); set(gca,'YTickLabel',{'-150','-75','0','75','150'},'FontSize',18);
xlabel('X (m)');ylabel('Y (m)');