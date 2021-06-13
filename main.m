function varargout = main(varargin)
% MAIN MATLAB code for main.fig
%      MAIN, by itself, creates a new MAIN or raises the existing
%      singleton*.
%
%      H = MAIN returns the handle to a new MAIN or the handle to
%      the existing singleton*.
%
%      MAIN('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MAIN.M with the given input arguments.
%
%      MAIN('Property','Value',...) creates a new MAIN or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before main_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to main_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @main_OpeningFcn, ...
                   'gui_OutputFcn',  @main_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

% --- Executes just before main is made visible.
function main_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to main (see VARARGIN)

% Choose default command line output for main
handles.output = hObject;

set(handles.pushbutton4,'String','Key Point Extraction');
set(handles.pushbutton5,'String','Feature Mapping');
set(handles.pushbutton6,'String','Hashing');
% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = main_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
clc;
global file_name;
global Ksegments;
global SLocations;
global SDescriptors;

[filename,pathname] = uigetfile('*.jpg;*.tif;*.png;*.gif;*.tiff','Select the image file');
file_name = filename;
% dimensione of statistics
Nb = [2, 8];
% number of cumulated bloks
Ns = 1;
bayer = [0, 1;
         1, 0];
global im_true;
im_true = imread(filename);
imshow(im_true)

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global file_name;
global Ksegments;

img = imread(file_name);
img=imresize(img,[256,256]);
iter = 50;

K = 7;
[segments, Cost7K] = KMeans(K,img,iter);
Ksegments = segments';
%% KM
%[Cost] = KMeans(K,img,iter)
% Selecting Image to Run
%{
[segments, Cost2K] = KMeans(2,img,iter);
%% K = 3
K = 3;
[segments, Cost3K] = KMeans(3,img,iter);
[segments, Cost4K] = KMeans(4,img,iter);
%% K = 5
K = 5;
[segments, Cost5K] = KMeans(5,img,iter);

[segments, Cost6K] = KMeans(6,img,iter);

%% K = 7
K = 7;
[segments, Cost7K] = KMeans(K,img,iter);
Ksegments = segments;

[segments, Cost8K] = KMeans(8,img,iter);

[segments, Cost9K] = KMeans(9,img,iter);

[segments, Cost10K] = KMeans(10,img,iter);


%% Cost plot
figure();
plot(Cost2K);
hold on; plot(Cost3K);
hold on; plot(Cost4K);
hold on; plot(Cost5K);
hold on; plot(Cost6K);
hold on; plot(Cost7K);
hold on; plot(Cost8K);
hold on; plot(Cost9K);
hold on; plot(Cost10K);


grid on;
xlabel('Iteration Count'); ylabel('Cost');
legend('K=2','K=3','K=4','K=5','K=6','K=7','K=8','K=9','K=10');
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in pushbutton5.
function pushbutton4_Callback(hObject, eventdata, handles)

global file_name;
global SLocations;
global SDescriptors;

%sift(im_true)
[Locations, Descriptors] = sift(imread(file_name));
display(size(Locations));
SLocations = Locations;

Descriptors = Descriptors';
Descriptors=double(Descriptors);
Descriptors=Descriptors./repmat(NormRow(Descriptors,2),1,128);

display("done keypoints");
SDescriptors = Descriptors;

% --- Executes on button press in pushbutton6.
function pushbutton5_Callback(hObject, eventdata, handles)

set(handles.pushbutton5,'String','Feature Mapping');

global file_name;
global Ksegments;
global SLocations;
global SDescriptors;

segments = Ksegments;
num_segments=max(segments(:));

Locations = SLocations;
Descriptors = SDescriptors;
MatchList = [];

% Assign a segment number to each keypoint
num_keypoint=size(Locations,1);
keypoint_segment=zeros(num_keypoint,1);
for k=1:num_keypoint
    keypoint_segment(k)=segments(round(Locations(k,1)),round(Locations(k,2)));
end
display("done 1");
for i=1:7
   idx1=find(keypoint_segment==i);
   if length(idx1)<2
       continue;
   end
   for j=1:length(idx1)
       D1 = Descriptors(idx1(j),:);
      for k=j+1:length(idx1)
        D2 = Descriptors(idx1(k),:);
        feat_dist=pdist2(D1,D2);
        if feat_dist < 0.5
            new_pairs=[idx1(j),idx1(k)];
            MatchList=[MatchList;new_pairs];
        end
      end
   end
   display(i);
end
display(MatchList);

%Show Match Keypoints
RGBimage=imread(file_name);
    figure;
    imshow(RGBimage);
    hold on;
    title('Match Keypoints');
    loc1_temp=Locations(MatchList(:,1),1:2);
    loc2_temp=Locations(MatchList(:,2),1:2);
    locs_temp=[loc1_temp;loc2_temp];
    plot(locs_temp(:,2),locs_temp(:,1),'mo');%Show points
    temp=[loc1_temp,loc2_temp,nan(size(loc2_temp,1),4)]';
    locs_temp=reshape(temp(:),4,[]);
    plot([locs_temp(2,:);locs_temp(4,:)],[locs_temp(1,:);locs_temp(3,:)],'b-');
    hold off;


%{
CC=zeros(num_segments*(num_segments-1)/2,1);
MatchList=[];
num=0;
for i=1:num_segments
    idx1=find(keypoint_segment==i);
    if length(idx1)<2
        num=num+num_segments-i;
        continue;
    end
    D1=Descriptors(idx1,:);
    min1=min(pdist(D1));
    for j=i+1:num_segments
        num=num+1;
        idx2=find(keypoint_segment==j);
        if length(idx2)<2
            continue;
        end
        D2=Descriptors(idx2,:);
        min2=min(pdist(D2));
        min3=min(min1,min2);
        feat_dist=pdist2(D1,D2)/min3;%Distances between descriptors
        [k1,k2]=find(feat_dist<=1/2);
        new_pairs=[idx1(k1),idx2(k2)];
        MatchList=[MatchList;new_pairs];
        CC(num)=size(new_pairs,1);
    end
end
num_match=size(MatchList,1);
%}
%{
global file_name;
I = imread(file_name);
% creating RGB matrices and storing dimensions
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);
sz = (R);
rows = sz(1,1);
columns = sz(1,2);
% Manipulating pixels
for i = 1:rows
    for j = 1:columns
         if B(i,j) <= 70;
            I(i,j,:) = 255;
        end
    end
end
figure, imshow(I)
%edge detection
I = rgb2gray(I);
I = imadjust(I, [0.3 0.9], [0 1]);
BW1 = edge(I,'canny',0.15);
figure, imshow(BW1);
[H,theta,rho] = hough(BW1);
% figure, imshow(imadjust(mat2gray(H)),[],'XData',theta,'YData',rho,...
%         'InitialMagnification','fit');
xlabel('\theta (degrees)'), ylabel('\rho');
axis on, axis normal, hold on;
colormap(hot) 
P = houghpeaks(H,75,'threshold',ceil(0.11*max(H(:))));
lines = houghlines(BW1,theta,rho,P,'FillGap',5,'MinLength',5);
figure, imshow(BW1), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',3,'Color','green');
end
%% Procedure for LHS
num_match = 7;
global file_name

clear I;
clear R;
clear G;
clear B;
%clear size;
I = imread(file_name);
%I = imread(fname);

% creating RGB matrices and storing dimensions
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);
sz = (R);
rows = sz(1,1);
columns = sz(1,2);
% Manipulating pixels
for i = 1:rows
    for j = 1:columns
         if B(i,j) <= 70;
            I(i,j,:) = 255;
        end
    end
end
figure, imshow(I)
%edge detection
I = rgb2gray(I);
I = imadjust(I, [0.3 0.9], [0 1]);
BW2 = edge(I,'canny',0.15);
figure, imshow(BW2);
[H,theta,rho] = hough(BW2);
% figure, imshow(imadjust(mat2gray(H)),[],'XData',theta,'YData',rho,...
%         'InitialMagnification','fit');
xlabel('\theta (degrees)'), ylabel('\rho');
axis on, axis normal, hold on;
colormap(hot)
P = houghpeaks(H,75,'threshold',ceil(0.11*max(H(:))));
lines = houghlines(BW2,theta,rho,P,'FillGap',5,'MinLength',5);
figure, imshow(BW2), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',3,'Color','green');
end
%% Finding angle of rotation by rotating by trial and error angles
diff = BW1 - BW2;
i=1;
%for i = 1:180
    BW1 = imrotate(BW1,i);
%    newdiff = BW1-BW2;
 %   if newdiff < diff
  %      diff = newdiff;
       angle = i;
 %   end
%end
%%%%%%%%%%%%%%%%%%%%%%%%
                    %I1=imread(file_name);
                    %I2=imread(file_name1);
% Get the Key Points
  Options.upright=true;
  Options.tresh=0.0001;
%  Ipts1=OpenSurf(I1,Options);
 % Ipts2=OpenSurf(I2,Options);
% Put the landmark descriptors in a matrix
  %D1 = reshape([Ipts1.descriptor],64,[]); 
  %D2 = reshape([Ipts2.descriptor],64,[]); 
% Find the best matches
  %err=zeros(1,length(Ipts1));
  %cor1=1:length(Ipts1); 
  %cor2=zeros(1,length(Ipts1));
  %for i=1:length(Ipts1),
   %   distance=sum((D2-repmat(D1(:,i),[1 length(Ipts2)])).^2,1);
    %  [err(i),cor2(i)]=min(distance);
 % end
% Sort matches on vector distance
%  [err, ind]=sort(err); 
%  cor1=cor1(ind); 
%  cor2=cor2(ind);
% Show both images
%  I = zeros([size(I1,1) size(I1,2)*2 size(I1,3)]);
 % I(:,1:size(I1,2),:)=I1; I(:,size(I1,2)+1:size(I1,2)+size(I2,2),:)=I2;
  %figure, imshow(I/255); hold on;
% Show the best matches
%  for i=1:30,
%      c=rand(1,3);
%      plot([Ipts1(cor1(i)).x Ipts2(cor2(i)).x+size(I1,2)],[Ipts1(cor1(i)).y Ipts2(cor2(i)).y],'-','Color',c)
%      plot([Ipts1(cor1(i)).x Ipts2(cor2(i)).x+size(I1,2)],[Ipts1(cor1(i)).y Ipts2(cor2(i)).y],'o','Color',c)
 % end
  %%%%%%%%%%%%%%%%%%%%%%%%%
                    %q=I1;
                    %p=I2;
  %[R,T] = icp(q,p,10);
  %%%%%%%%%%%%%%%%%%%%%%%%%
  m = 80; % width of grid
n = m^2; % number of points

[X,Y] = meshgrid(linspace(-2,2,m), linspace(-2,2,m));

X = reshape(X,1,[]);
Y = reshape(Y,1,[]);

Z = sin(X).*cos(Y);

% Create the data point-matrix
D = [X; Y; Z];

% Translation values (a.u.):
Tx = 0.5;
Ty = -0.3;
Tz = 0.2;

% Translation vector
T = [Tx; Ty; Tz];

% Rotation values (rad.):
rx = 0.3;
ry = -0.2;
rz = 0.05;

Rx = [1 0 0;
      0 cos(rx) -sin(rx);
      0 sin(rx) cos(rx)];
  
Ry = [cos(ry) 0 sin(ry);
      0 1 0;
      -sin(ry) 0 cos(ry)];
  
Rz = [cos(rz) -sin(rz) 0;
      sin(rz) cos(rz) 0;
      0 0 1];

% Rotation matrix
R = Rx*Ry*Rz;

% Transform data-matrix plus noise into model-matrix 
M = R * D + repmat(T, 1, n);

% Add noise to model and data
rng(2912673);
M = M + 0.01*randn(3,n);
D = D + 0.01*randn(3,n);

%% Run ICP (standard settings)
[Ricp Ticp ER t] = icp(M, D, 15);

% Transform data-matrix using ICP result
Dicp = Ricp * D + repmat(Ticp, 1, n);

% Plot model points blue and transformed points red
figure;
subplot(2,2,1);
plot3(M(1,:),M(2,:),M(3,:),'bo',D(1,:),D(2,:),D(3,:),'r.');
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title('Red: z=sin(x)*cos(y), blue: transformed point cloud');

% Plot the results
subplot(2,2,2);
plot3(M(1,:),M(2,:),M(3,:),'bo',Dicp(1,:),Dicp(2,:),Dicp(3,:),'r.');
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title('ICP result');

% Plot RMS curve
subplot(2,2,[3 4]);
plot(0:15,ER,'--x');
xlabel('iteration#');
ylabel('d_{RMS}');
legend('bruteForce matching');
title(['Total elapsed time: ' num2str(t(end),2) ' s']);

%% Run ICP (fast kDtree matching and extrapolation)
[Ricp Ticp ER t] = icp(M, D, 15, 'Matching', 'kDtree', 'Extrapolation', true);

% Transform data-matrix using ICP result
Dicp = Ricp * D + repmat(Ticp, 1, n);

% Plot model points blue and transformed points red
figure;
subplot(2,2,1);
plot3(M(1,:),M(2,:),M(3,:),'bo',D(1,:),D(2,:),D(3,:),'r.');
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title('Red: z=sin(x)*cos(y), blue: transformed point cloud');

% Plot the results
subplot(2,2,2);
plot3(M(1,:),M(2,:),M(3,:),'bo',Dicp(1,:),Dicp(2,:),Dicp(3,:),'r.');
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title('ICP result');

% Plot RMS curve
subplot(2,2,[3 4]);
plot(0:15,ER,'--x');
xlabel('iteration#');
ylabel('d_{RMS}');
legend('kDtree matching and extrapolation');
title(['Total elapsed time: ' num2str(t(end),2) ' s']);

%% Run ICP (partial data)

% Partial model point cloud
Mp = M(:,Y>=0);

% Boundary of partial model point cloud
b = (abs(X(Y>=0)) == 2) | (Y(Y>=0) == min(Y(Y>=0))) | (Y(Y>=0) == max(Y(Y>=0)));
bound = find(b);

% Partial data point cloud
Dp = D(:,X>=0);

[Ricp Ticp ER t] = icp(Mp, Dp, 50, 'EdgeRejection', true, 'Boundary', bound, 'Matching', 'kDtree');

% Transform data-matrix using ICP result
%Dicp = Ricp * Dp + repmat(Ticp, 1, size(Dp,2));

% Plot model points blue and transformed points red
figure;
subplot(2,2,1);
plot3(Mp(1,not(b)),Mp(2,not(b)),Mp(3,not(b)),'bo',...
      Mp(1,b),Mp(2,b),Mp(3,b),'go',...
      Dp(1,:),Dp(2,:),Dp(3,:),'r.')
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title('Red: z=sin(x)*cos(y), blue: transformed point cloud');

% Plot the results
subplot(2,2,2);
plot3(Mp(1,not(b)),Mp(2,not(b)),Mp(3,not(b)),'bo',...
      Mp(1,b),Mp(2,b),Mp(3,b),'go',...
      Dicp(1,:),Dicp(2,:),Dicp(3,:),'r.');
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title('ICP result');

% Plot RMS curve
subplot(2,2,[3 4]);
plot(0:50,ER,'--x');
xlabel('iteration#');
ylabel('d_{RMS}');
legend('partial overlap');
title(['Total elapsed time: ' num2str(t(end),2) ' s']);
%}
if num_match==0
    disp(' The Image Is Not Forged..!!');
else
    disp(' The Image is Forged..!!');
end
global file_name;
%extra('flowers-tampered.tiff');


% --- Executes on button press in pushbutton4.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global file_name;
image = imread(file_name); % Read in the image
image = imresize(image,[512,512]);
hashing(image);

disp('completted')



