%% Import Script for EBSD Data
%
% This script was automatically created by the import wizard. You should
% run the whoole script or parts of it in order to import your data. There
% is no problem in making any changes to this script.

%% Specify Crystal and Specimen Symmetries

% crystal symmetry
CS = {... 
  'notIndexed',...
  crystalSymmetry('6/mmm', [3 3 4.7], 'X||a*', 'Y||b', 'Z||c*', 'mineral', 'Ti-alpha', 'color', [0.53 0.81 0.98]),...
  crystalSymmetry('m-3m', [3.2 3.2 3.2], 'mineral', 'Ti-beta', 'color', [0.56 0.74 0.56])};

% plotting convention
setMTEXpref('xAxisDirection','east');
setMTEXpref('zAxisDirection','inToPlane');

%% Specify File Names

% path to files
pname = 'C:\Users\Phllip\Downloads';

% which files to be imported
fname = [pname '\Test Cantilever 1 Specimen 1 Site 1 Map Data 5.ctf'];

%% Import the Data

% create an EBSD variable containing the data
ebsd = EBSD.load(fname,CS,'interface','ctf',...
  'convertEuler2SpatialReferenceFrame');

%% Define IPF key
ebsd_Ti_alpha = ebsd('Ti-alpha');

% this defines an ipf color key for the Forsterite phase
ipfKey = ipfHSVKey(ebsd_Ti_alpha);
plot(ipfKey)

%% Prepare data
%
% For Aztec EBSD data:
% bc = band contrast
% bs = band slope (Microsc. Microanal. 19, S5, 13–16, 2013)
% bands = # indexed bands
% mad = mean angular deviation ("The mean angular deviation (MAD) indicates
% the misfit between the measured and the calculated angles between bands,
% so the larger the MAD value, the higher the misfit and the lower the
% indexing confidence. It is also a good indicator of lattice strain or
% defect density" -
% https://www.researchgate.net/post/How_is_image_quality_quantified_in_EBSD_analysis)

% 1) select data

% 2) remove low bc

% 3) calc grains

% 4) calc ODF

% 5) plot pole figures

% remove points with low IQ (edge of specimen)
bc = ebsd_Ti_alpha.bc;
histogram(bc)

bs = ebsd_Ti_alpha.bs;
histogram(bs)


% compute the colors
ipfKey.inversePoleFigureDirection = zvector;
colour = ipfKey.orientation2color(ebsd_Ti_alpha.orientations);


plot(ebsd('Ti-alpha'),colour)

grainAngleThreshold = 10; % degrees
fprintf('Calculating grains with %2.0f degree threshold...', grainAngleThreshold)
[grains, ebsd_Ti_alpha.grainId] = calcGrains(ebsd_Ti_alpha, 'angle', grainAngleThreshold*degree);
fprintf(' done.\n')
hold on
plot(grains,'linewidth',2)
hold off

histogram(grains.area, 100);

minGrainAreaThreshold = 100;
fprintf('Removing grains with area < %i pixels...', minGrainAreaThreshold)
grainsToRemove = grains(grains.area < minGrainAreaThreshold);
% ebsd_Ti_alpha(grainsToRemove).phaseId = 0;
ebsd_Ti_alpha(grainsToRemove) = [];
fprintf(' done.\n')


% compute the colors
ipfKey.inversePoleFigureDirection = zvector;
colour = ipfKey.orientation2color(ebsd_Ti_alpha.orientations);

plot(ebsd_Ti_alpha, colour)
