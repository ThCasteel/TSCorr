% Step 1: Plot gradient magnitude as background
figure;
imagesc(x, y, gradMag); 
axis xy equal;
colormap(parula); 
colorbar;
xlabel('Easting (m)');
ylabel('Northing (m)');
title('Gradient Magnitude with Fault Traces');
hold on;

% Step 2: Overlay fault traces
jsonData = fileread('FaultMerge.geojson');
geo = jsondecode(jsonData);
coords = geo.features(1).geometry.coordinates;

xf = []; yf = [];
for i = 1:length(coords)
    segment = coords{i};
    xf = [xf; segment(:,1); NaN];
    yf = [yf; segment(:,2); NaN];
end

plot(xf, yf, 'k-', 'LineWidth', 1.5);
