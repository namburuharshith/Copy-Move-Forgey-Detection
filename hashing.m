function hashing(rgbImage)

forgered = 0;

[rows columns numberOfColorBands] = size(rgbImage)
%==========================================================================
% The first way to divide an image up into blocks is by using mat2cell().
blockSizeR = 2; % Rows in block.
blockSizeC = 2; % Columns in block.
% Figure out the size of each block in rows. 
% Most will be blockSizeR but there may be a remainder amount of less than that.
wholeBlockRows = floor(rows / blockSizeR);
blockVectorR = [blockSizeR * ones(1, wholeBlockRows), rem(rows, blockSizeR)];
% Figure out the size of each block in columns. 
wholeBlockCols = floor(columns / blockSizeC);
blockVectorC = [blockSizeC * ones(1, wholeBlockCols), rem(columns, blockSizeC)];
% Create the cell array, ca.  
% Each cell (except for the remainder cells at the end of the image)
% in the array contains a blockSizeR by blockSizeC by 3 color array.
% This line is where the image is actually divided up into blocks.
if numberOfColorBands > 1
  % It's a color image.
  ca = mat2cell(rgbImage, blockVectorR, blockVectorC, numberOfColorBands);
else
  ca = mat2cell(rgbImage, blockVectorR, blockVectorC);
end
% Now display all the blocks.
plotIndex = 1;
numPlotsR = size(ca, 1);
numPlotsC = size(ca, 2);
hash=string.empty;
for r = 1 : numPlotsR-1
  for c = 1 : numPlotsC-1
    rgbBlock = ca{r,c};
    
    [rowsB columnsB numberOfColorBandsB] = size(rgbBlock); 
    
    if c==2
       continue; 
    end
    image = rgbBlock;
    [m, n, ~] = size(image);       % Gives rows, columns, ignores number of channels
% Starts by separating the image into RGB channels
flat_R = reshape(image(:,:,1)',[1 m*n]); % Reshapes Red channel matrix into a 1 by m*n uint8 array
flat_G = reshape(image(:,:,2)',[1 m*n]); % 
flat_B = reshape(image(:,:,3)',[1 m*n]); % 
flat_RGB = [flat_R, flat_G, flat_B];     % Concatenates all RGB vals, into one long 1 by 3*m*n array
string_RGB = num2str(flat_RGB);                         % Converts numeric matrices to a string
string_RGB = string_RGB(~isspace(num2str(string_RGB))); % Removes spaces - though this is not strictly necessary I think
% Perform hashing
sha256hasher = System.Security.Cryptography.SHA256Managed;           % Create hash object (?) - this part was copied from the forum post mentioned above, so no idea what it actually does
imageHash_uint8 = uint8(sha256hasher.ComputeHash(uint8(string_RGB))); % Find uint8 of hash, outputs as a 1x32 uint8 array
imageHash_hex = dec2hex(imageHash_uint8) % Convert uint8 to hex, if necessary. This step is optional depending on your application.
hash(r) = num2str(reshape(imageHash_hex', 1, []));

  end
end

for i = 1 : 256
  for j = i+1 : 256
       
    if hash(i) == hash(j)
      forgered = 1; 
    end
        
   end
end

display(forgered);

if forgered == 1 
    
    f = msgbox('Forgery Detected');
else
    
    f = msgbox('Forgery not Detected');
end


end