a = [1,2,3];
5+6;
1~=2;
1||0;
1&&0;
xor(1,0);
a=pi;
disp(a);
disp(sprintf('2 decimals: %0.2f', a));
format long;
a
format short;
a
A = [1 2;3 4;5 6];
%pinv(A);
v=0:0.1:2;
e=ones(2,3);
rand(3,3);
%w = -6 + sqrt(10)*randn(1,1000);
%hist(w);
I=eye(4);
%help('rand');

size(A,1);
load featuresX.dat;
load('priceY.dat');
size(featuresX);
%who;
%whos;
clear('e');
%v = priceY(1:3);
%save hello.m v;
A(3,2);
A(2,:);A(:,2);A([1 3],:);
A = [A, [100;101;102]];
A(2,:) = [10,11,12];
A = [1 2;3 4;5 6];
B = [11 12; 13,14; 15,16];
%C = [A,B];
%C = [A;B];
C=[1 2; 3 4];
A*C;
A.*B;
1./A;
log(A); exp(A);-A;A + ones(3,2); A + 1; A';A'';
[val, ind] = max(A);
find(A<3);
R = randn(3);
ceil(R);
A = magic(3);
max(A,[], 1);
max(A,[], 2);
max(max(A));max(A(:));
magic(9).*eye(9);
sum(sum(magic(9).*eye(9)));
flipud(eye(9));
A*pinv(A);
%plot(x,y1(x));
%hold on;
%plot(x, y2(x));
%legend, xlabel, ylabel, title...
%print -dpng 'myPlot.png' %save the plot
%close %close the plot
%figure(1) plot...; figure(2) plot...
%subplot(1,2,1) plot...; subplot(1,2,2) plot...
%axis([-1 0 1]) %set the axis values
%A = magic(5);
%imagesc(A);
%imagesc(A), colorbar, colormap gray;
%print -dpng 'myPlot.png' %save the plot
%for i=1:10 do something... end;
%if elseis else end;
X = [1 1 1; 1 2 3];
y = [1 2 3];
theta = [0; 1];
J = costFunction(X, y, theta);
theta = [0; 0];
J = costFunction(X, y, theta);
theta = gradientDescent(X, y, theta, 0.5)