%% Gradient-percentile sweep  ---------------------------------------------
% --- 1.  Load time-series displacement -----------------------------------
S   = load('utm37n_41_02_1_1_02_01.mat');   % ratex, ratey, xout, yout
Ux  = S.ratex;                              % east-west disp.  (m)
Uy  = S.ratey;                              % north-south disp. (m)

% --- 2.  Grid setup -------------------------------------------------------
x  = S.xout(:);                 % 1-D easting vector  (UTM m)
y  = S.yout(:);                 % 1-D northing vector (UTM m)
dx = median(diff(x));           % pixel size (m)
dy = abs(median(diff(y)));
[Xg,Yg] = meshgrid(x,y);

%% --- 3.  Ensure Y axis is north-down (row 1 = north) --------------------
fprintf('Original y(1)=%.0f   y(end)=%.0f\n', y(1), y(end));
if y(2) > y(1)
    fprintf('y ascending → flipping rasters so row 1 = north\n');
    Ux = flipud(Ux);  Uy = flipud(Uy);
    y  = flipud(y);   [Xg,Yg] = meshgrid(x,y);
end
fprintf('Now     y(1)=%.0f   y(end)=%.0f\n\n', y(1), y(end));

%% --- 4.  Gradient magnitude --------------------------------------------
[Ux_x,Ux_y] = gradient(Ux, dx, dy);
[Uy_x,Uy_y] = gradient(Uy, dx, dy);
gradMag     = hypot(Ux_x + Uy_y,  Ux_y - Uy_x);

%% --- 5.  Load merged fault (GeoJSON) & densify --------------------------
geo    = jsondecode(fileread('FaultMerge.geojson'));
segments = geo.features(1).geometry.coordinates;      % may be mixed

v = [];                                               % vertices Nx2
for s = 1:numel(segments)
    elem = segments{s};
    if iscell(elem)           % still wrapped once more
        seg = cell2mat(elem');  % Nx2 double
    else                       % already numeric
        seg = elem;            % Nx2 double
    end
    v = [v; seg];                                %#ok<AGROW>
end


% keep only vertices inside raster extent (optional tidy step)
inbox = v(:,1)>=min(x) & v(:,1)<=max(x) & ...
        v(:,2)>=min(y) & v(:,2)<=max(y);
v = v(inbox,:);

% ---------- densify each strand separately --------------------------------
step = 500;                      % metres
vdense = [];                      % will collect all strands with NaN breaks

for s = 1:numel(segments)
    % unwrap numeric matrix for this strand
    if iscell(segments{s}), seg = cell2mat(segments{s}'); else, seg = segments{s}; end

    % option: discard points outside raster bbox
    inbox = seg(:,1)>=min(x) & seg(:,1)<=max(x) & ...
            seg(:,2)>=min(y) & seg(:,2)<=max(y);
    seg = seg(inbox,:);

    % densify current strand
    vx = seg(:,1);  vy = seg(:,2);
    for i = 1:size(seg,1)-1
        vec = seg(i+1,:) - seg(i,:);
        L   = norm(vec);
        n   = max(floor(L/step)-1,0);
        if n
            t  = (1:n)'*step/L;
            vx = [vx; seg(i,1)+t*vec(1)]; %#ok<AGROW>
            vy = [vy; seg(i,2)+t*vec(2)]; %#ok<AGROW>
        end
    end

    vdense = [vdense; [vx vy]; NaN NaN];   %#ok<AGROW> % add NaN break
end

% drop final NaN row
vdense = vdense(~any(isnan(vdense),2),:);


%% --- 6.  Rasterise fault, compute pixel-to-fault distance (km) ----------
mask  = false(size(Xg));
col   = round( (vdense(:,1)-x(1))/dx ) + 1;
row   = round( (y(1)-vdense(:,2))/dy ) + 1;
valid = col>=1 & col<=numel(x) & row>=1 & row<=numel(y);
mask(sub2ind(size(mask), row(valid), col(valid))) = true;

dist_km = bwdist(mask) * dx / 1000;

%% --- 7.  Buffer sweep & gradient percentiles ----------------------------
buffers = [2 3 5 10 15];                       % km
fprintf('dx = %.2f m   dy = %.2f m\n', dx, dy);
for buf = buffers
    far = gradMag(dist_km > buf);
    fprintf('%2dkm  n=%d  95th=%.2f%%  50th=%.2f%%  5th=%.2f%%\n', ...
            buf, numel(far), prctile(far,95)*100, ...
            prctile(far,50)*100, prctile(far,5)*100);
end

%% --- 8.  Chosen buffer --------------------------------------------------------------
bufChoice = 3;                              % km
far       = gradMag(dist_km > bufChoice);
p95 = prctile(far,95);  p50 = prctile(far,50);  p05 = prctile(far,5);

fprintf('\nChosen buffer = %g km | Gradient thresholds:  >%.2f%%  |  %.2f–%.2f%%  |  <%.2f%%\n', ...
        bufChoice, p95*100, p50*100, p95*100, p05*100);
fprintf('min(dist_km)=%.2f  1%%-tile=%.2f  median=%.2f\n\n', ...
        min(dist_km(:)), prctile(dist_km(:),1), prctile(dist_km(:),50));

%% --- 9.  Quick visual check ---------------------------------------------
figure
imagesc('XData',x,'YData',y,'CData',dist_km<5);
axis xy equal; colormap(gray)
title('< 5-km mask (white)'); hold on
plot(vdense(:,1), vdense(:,2),'r-','LineWidth',1);
