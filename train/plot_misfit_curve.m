clc;clear;close all;
%% load data
a=load('./Loss.log.txt');
train_acc=a(:,1);
train_loss=a(:,2);
val_acc=a(:,3);
val_loss=a(:,4);
%%
figure(1);
plot(train_acc,'-*','color','red','linewidth',2);hold on;
plot(train_loss,'-o','color','blue','linewidth',2);hold on;
plot(val_acc,'-*','color','green','linewidth',2);hold on;
plot(val_loss,'-o','color','k','linewidth',2);hold on;
set(gcf,'unit','centimeters','position',[0 0 15 12]);
set(gca,'FontSize',16);
set(gca,'XAxisLocation','bottom','YAxisLocation','left');
axis([1 50 0 1]);hold on
set(gca,'Xtick',[0 10 20 30 40 50]);
set(gca,'XTickLabel',{'0','10','20','30','40','50'});
h=legend('Training Accuracy','Training Loss','Validation Accuracy','Validation Loss');
set(h,'Fontsize',18);
xlabel('Epoch Number','fontsize',18);
ylabel('Normalized Accuracy-Loss','fontsize',18);