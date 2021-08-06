function plot_cfmatrix(cfmat, classname, vfmt, cmap)

imagesc(cfmat);  % Create a colored plot of the matrix values
if strcmp(cmap, 'gray')
    colormap(flipud(gray));  % Change the colormap to gray (so higher values are black and lower values are white)
else
    colormap(cmap);
end
textStrings = num2str(cfmat(:), vfmt);  % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
 
%% ## New code: ###
%idx = find(strcmp(textStrings(:), '0.00'));
%textStrings(idx) = {'   '};
%% -----------------------------
 % Create x and y coordinates for the strings
[x, y] = meshgrid(1:5);   
% Plot the strings
hStrings = text(x(:), y(:), textStrings(:),...      
                'HorizontalAlignment','center');
% Get the middle value of the color range
midValue = mean(get(gca,'CLim'));  

% Choose white or black for the  text color of the strings so
textColors = repmat(cfmat(:) > midValue, 1, 3);  
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings, {'Color'}, num2cell(textColors, 2));  %# Change the text colors
 
set(gca,'XTick',1:5,...                         %# Change the axes tick marks
        'XTickLabel', classname,...  %#   and tick labels
        'YTick',1:5,...
        'YTickLabel', classname,...
        'TickLength',[0 0]);
