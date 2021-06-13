function [imseg, Cost] = KMeans(K,img,iter)
%%% K means 
%   K    - Number of Clusters
%   img  - Input Image in RGB color space
%   iter - Maximum Number of Iterations
%%%

% img = imread('peppers.bmp');
% K =3;
% iter = 25;

%data = gpuArray( reshape(img,[ size(img,1)*size(img,2), 3]) );
display(img);
data = ( reshape(img,[ size(img,1)*size(img,2), 3]) );

%[cluster_index, cluster_centre] = my_kmeans(data,numClusters,5);


%% Kmeans Start

% Get data sizes
NRows = size(data,1);
NCols = size(data,2);

%% Initializing Centres
if exist('Centre','Var')
    clear Centre;
end

for i = 1 : K
    Centre (i,:) = randi([0 255],1,3);
end

% Setting initial cluster values
ClusterNumberOld = zeros(NRows,1);

%% Setting up segmented plot
imseg = zeros (size(img,1),size(img,2));

CostDiff = 0;
S = repmat([1],numel(imseg),1);
s = S(:,1);
z = reshape(imseg,size(img,1)*size(img,2),1);
Z = [z z z];
Cost = 0;

figure();

%Original Image
subplot(2,2,1);
imshow(img);
title('Original Image');

% 3D plot of clusters in color space
p1 = subplot(2,2,3);
scatter3(p1,data(:,1),data(:,2),data(:,3),s,Z);
title('clusters in color space');
grid on;
xlabel('R'); ylabel('G'); zlabel('B');

% Segmented Image
subplot(2,2,2);
p2 = imagesc(imseg);
title(sprintf('Segmented Image; K = %d ; Iteration = 1', K));

% Cost
p3 = subplot(2,2,4);
plot(p3,Cost);
grid on;
title(sprintf('Cost; K = %d ; Iteration = 1', K));
xlabel('Iteration'); ylabel('Cost');


%% Loop start
for i = 1:iter
    fprintf('K Iteration --> %d \t', i);
    clear Dist;
    % Finding Eucledian Dist
    Dist =  pdist2(double(data),Centre);
    
    % Sorting & assignign nearest cluster
    [DistShortest , ClusterNumber] = min(Dist,[],2);
    
    Cost(i) = cSum(gather(DistShortest), K); 
    fprintf('Cost --> %d\n', Cost(i));
    
    
    % Updatinf Cluster Centres
    for j = 1 : K
        ClusterAssignments = find(ClusterNumber == j) ;
        
        % For all assgined values of assgined clusters
        if gather(ClusterAssignments)
            Centre(j,:) = mean( gather(data( ClusterAssignments,:)) );
        end
        
    end
    
    Op = [data , ClusterNumber];
    
    
    for k=1:max( gather(ClusterNumber) )
        imseg( gather(ClusterNumber) ==k)=k;
    end
    
    % Visualise Image
%    z = reshape(imseg,512*512,1);
 %   z = z / K;
    %z1 = dec2bin(z,3);
    %z2 = [str2num(z1(:,1)) str2num(z1(:,2)) str2num(z1(:,3))];
    Z = [1-z z z];
    
    % Updating Plots
    pause(0.1);
    subplot(2,2,3)
    scatter3(p1,data(:,1),data(:,2),data(:,3),s,Z); 
    title('clusters in color space');
    xlabel('R'); ylabel('G'); zlabel('B');
    
    subplot(2,2,2)
    set(p2, 'CData', imseg);
    title(sprintf('Segmented Image; K = %d ; Iteration = %d', K,i));
    
    subplot(2,2,4)
    plot(p3,Cost); 
    title(sprintf('Cost; K = %d ; Iteration = %d', K, i));
    grid on;
    xlabel('Iteration'); ylabel('Cost');
    
    % Iteration Stop COndition
    
    if i >= 2
        CostDiff(i) = Cost(i-1) - Cost(i);
        if CostDiff(i) < 10
            break;
        end
    end
    
end


end