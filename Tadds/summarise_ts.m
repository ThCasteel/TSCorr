function summarise_ts(matfile, outfile)
% summarise_ts  Global stats + histograms for deformation products
%
%   summarise_ts("utm37n_41_02_1_1_02_01.mat")
%   summarise_ts("utm37n_41_02_1_1_02_01.mat","PlanetNonCorrectionTimeseries.csv")
%
% If OUTFILE is omitted, the default
%   "PlanetNonCorrectionTimeseries.csv" is used.
%
%   • Saves summary statistics (one row per variable) to the CSV.
%   • Prints the same stats to the Command Window.
%   • Keeps the original histograms for 2-D maps.

% -------------------------------------------------------------------------
arguments
    matfile (1,1) string {mustBeFile}
    outfile  (1,1) string = "PlanetNonCorrectionTimeseries.csv"
end

% ---- Load & inspect -----------------------------------------------------
S  = load(matfile);
fn = string(fieldnames(S));

fprintf('Loaded "%s"\n', matfile);
fprintf('Variables:  %s\n', strjoin(fn, ', '));

target  = ["rate","ratex","ratey","ratestd","ratexstd","rateystd"];
present = intersect(target, fn);

% Pre-allocate a structure array for later conversion to table
stats = struct('Variable',     string.empty ...
              ,'N',            [] ...
              ,'Mean',         [] ...
              ,'Median',       [] ...
              ,'Std',          [] ...
              ,'Min',          [] ...
              ,'Max',          []);

% ---- Loop over requested variables --------------------------------------
for k = 1:numel(present)
    name = present(k);
    A    = S.(name);

    % Flatten & discard NaNs (faster than ismissing for numeric)
    vec  = A(:);
    vec  = vec(~isnan(vec));

    % Compute stats once
    N      = numel(vec);
    mu     = mean(vec);
    med    = median(vec);
    sigma  = std(vec);   % unbiased by default
    lo     = min(vec);
    hi     = max(vec);

    % Echo to command window
    fprintf('\n%s  (N = %d):\n', name, N);
    fprintf('   mean     %8.4f\n', mu);
    fprintf('   median   %8.4f\n', med);
    fprintf('   std-dev  %8.4f\n', sigma);
    fprintf('   min      %8.4f\n', lo);
    fprintf('   max      %8.4f\n', hi);

    % Stash in struct array
    stats(k) = struct('Variable', name , 'N', N , ...
                      'Mean', mu , 'Median', med , ...
                      'Std',  sigma , 'Min', lo , 'Max', hi);

    % Histogram
    if ismatrix(A) && ~isvector(A)
        figure('Name',name);
        histogram(vec,50,'EdgeColor','none');
        title(sprintf('Histogram of %s', name),'Interpreter','none');
        xlabel(name); ylabel('Frequency');
    end
end

% ---- Write CSV ----------------------------------------------------------
T = struct2table(stats);
writetable(T, outfile);           % saves in current folder unless path given
fprintf('\nSummary written to "%s"\n', outfile);
end
