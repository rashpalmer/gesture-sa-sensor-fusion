function model = ml_train_baseline(labeled_data, params)
%ML_TRAIN_BASELINE Train a baseline ML classifier for gesture recognition
%
%   model = ML_TRAIN_BASELINE(labeled_data, params)
%
%   Trains a machine learning classifier (kNN, SVM, or Decision Tree) on
%   labeled gesture data. This provides a path toward ML-based classification
%   that can be extended to deep learning.
%
%   INPUTS:
%       labeled_data - Can be one of:
%           (a) Path to .mat file containing labeled examples
%           (b) Path to .csv file with features and labels
%           (c) Struct array with fields:
%               .features : NxM feature matrix (N examples, M features)
%               .labels   : Nx1 cell array of gesture label strings
%               .names    : 1xM cell array of feature names (optional)
%           (d) Table with feature columns and 'label' column
%
%       params - Configuration from config_params.m (optional)
%                Key fields:
%                  params.ml.method : 'knn', 'svm', or 'tree' (default: 'knn')
%                  params.ml.k      : k for kNN (default: 5)
%                  params.ml.kernel : SVM kernel type (default: 'rbf')
%                  params.ml.standardize : true/false (default: true)
%
%   OUTPUTS:
%       model - Struct containing:
%               .classifier   : Trained MATLAB classifier object
%               .method       : String identifying method used
%               .feature_names: Cell array of feature names
%               .class_names  : Cell array of gesture labels
%               .mu           : Feature means (for standardization)
%               .sigma        : Feature stds (for standardization)
%               .train_accuracy: Training accuracy
%               .confusion_matrix: Training confusion matrix
%               .cv_accuracy  : Cross-validation accuracy (if computed)
%               .timestamp    : Training timestamp
%
%   EXPECTED DATA FORMAT (CSV):
%   ----------------------------
%   duration,gyr_rms_x,gyr_rms_y,gyr_rms_z,...,label
%   0.52,1.23,0.45,2.10,...,twist
%   0.71,2.50,1.80,0.30,...,flip_up
%   ...
%
%   EXPECTED DATA FORMAT (MAT):
%   ---------------------------
%   labeled_gestures.features = [N x M double]
%   labeled_gestures.labels = {'twist'; 'shake'; 'flip_up'; ...}  % Nx1 cell
%   labeled_gestures.feature_names = {'duration', 'gyr_rms_x', ...} % 1xM cell
%
%   USAGE EXAMPLE:
%       % Train from labeled data file
%       params = config_params();
%       model = ml_train_baseline('data/labeled/gestures_labeled.mat', params);
%
%       % Train from feature matrix directly
%       data.features = rand(100, 20);  % 100 examples, 20 features
%       data.labels = repmat({'twist'; 'shake'; 'flip'; 'push'; 'circle'}, 20, 1);
%       model = ml_train_baseline(data, params);
%
%       % Save model
%       save('models/gesture_model.mat', 'model');
%
%   See also: ML_PREDICT_BASELINE, FITCKNN, FITCSVM, FITCTREE, EXTRACT_FEATURES
%
%   MathWorks Documentation:
%   - fitcknn: https://www.mathworks.com/help/stats/fitcknn.html
%   - fitcsvm: https://www.mathworks.com/help/stats/fitcsvm.html
%   - fitctree: https://www.mathworks.com/help/stats/fitctree.html
%   - Classification Learner App: https://www.mathworks.com/help/stats/classification-learner-app.html
%
%   Author: Claude (Anthropic) - Sensor Fusion Demonstrator
%   Date: January 2026

    %% Input validation and parameter setup
    if nargin < 1 || isempty(labeled_data)
        error('ml_train_baseline:NoInput', ...
              'Labeled data required. Provide file path or data struct.');
    end
    
    if nargin < 2 || isempty(params)
        params = config_params();
    end
    
    % Get ML parameters with defaults
    if isfield(params, 'ml')
        ml_params = params.ml;
    else
        ml_params = struct();
    end
    
    method = getFieldOrDefault(ml_params, 'method', 'knn');
    k_neighbors = getFieldOrDefault(ml_params, 'k', 5);
    svm_kernel = getFieldOrDefault(ml_params, 'kernel', 'rbf');
    do_standardize = getFieldOrDefault(ml_params, 'standardize', true);
    do_crossval = getFieldOrDefault(ml_params, 'cross_validate', true);
    cv_folds = getFieldOrDefault(ml_params, 'cv_folds', 5);
    verbose = getFieldOrDefault(params, 'verbose', true);
    
    %% Load and parse labeled data
    if verbose
        fprintf('\n=== ML Training: %s Classifier ===\n', upper(method));
    end
    
    [features, labels, feature_names] = parse_labeled_data(labeled_data, verbose);
    
    % Validate dimensions
    [n_samples, n_features] = size(features);
    
    if length(labels) ~= n_samples
        error('ml_train_baseline:DimensionMismatch', ...
              'Number of labels (%d) must match number of feature rows (%d).', ...
              length(labels), n_samples);
    end
    
    if verbose
        fprintf('Data loaded: %d samples, %d features\n', n_samples, n_features);
    end
    
    % Get unique classes
    unique_labels = unique(labels);
    n_classes = length(unique_labels);
    
    if verbose
        fprintf('Classes: %s\n', strjoin(unique_labels, ', '));
        
        % Print class distribution
        fprintf('Class distribution:\n');
        for i = 1:n_classes
            count = sum(strcmp(labels, unique_labels{i}));
            fprintf('  %s: %d (%.1f%%)\n', unique_labels{i}, count, 100*count/n_samples);
        end
    end
    
    %% Handle missing/invalid features
    % Replace NaN with column mean
    for j = 1:n_features
        col = features(:, j);
        nan_idx = isnan(col);
        if any(nan_idx)
            col_mean = mean(col(~nan_idx));
            if isnan(col_mean)
                col_mean = 0;  % All NaN column
            end
            features(nan_idx, j) = col_mean;
            if verbose
                fprintf('Warning: Replaced %d NaN values in feature %d with mean\n', ...
                        sum(nan_idx), j);
            end
        end
    end
    
    % Replace Inf with large finite values
    features(isinf(features) & features > 0) = 1e10;
    features(isinf(features) & features < 0) = -1e10;
    
    %% Standardize features (z-score normalization)
    if do_standardize
        mu = mean(features, 1);
        sigma = std(features, 0, 1);
        sigma(sigma < 1e-10) = 1;  % Avoid division by zero
        features_norm = (features - mu) ./ sigma;
        
        if verbose
            fprintf('Features standardized (z-score normalization)\n');
        end
    else
        mu = zeros(1, n_features);
        sigma = ones(1, n_features);
        features_norm = features;
    end
    
    %% Train classifier based on selected method
    switch lower(method)
        case 'knn'
            % k-Nearest Neighbors
            if verbose
                fprintf('Training kNN classifier (k=%d)...\n', k_neighbors);
            end
            
            classifier = fitcknn(features_norm, labels, ...
                'NumNeighbors', k_neighbors, ...
                'Distance', 'euclidean', ...
                'Standardize', false, ...  % Already standardized
                'ClassNames', unique_labels);
            
        case 'svm'
            % Support Vector Machine (multiclass via ECOC)
            if verbose
                fprintf('Training SVM classifier (kernel=%s)...\n', svm_kernel);
            end
            
            if n_classes == 2
                % Binary SVM
                classifier = fitcsvm(features_norm, labels, ...
                    'KernelFunction', svm_kernel, ...
                    'Standardize', false, ...
                    'ClassNames', unique_labels);
            else
                % Multiclass SVM using Error-Correcting Output Codes
                template = templateSVM('KernelFunction', svm_kernel, ...
                                       'Standardize', false);
                classifier = fitcecoc(features_norm, labels, ...
                    'Learners', template, ...
                    'ClassNames', unique_labels);
            end
            
        case 'tree'
            % Decision Tree
            if verbose
                fprintf('Training Decision Tree classifier...\n');
            end
            
            classifier = fitctree(features_norm, labels, ...
                'ClassNames', unique_labels, ...
                'MinLeafSize', max(1, floor(n_samples / 50)));
            
        otherwise
            error('ml_train_baseline:UnknownMethod', ...
                  'Unknown method: %s. Use ''knn'', ''svm'', or ''tree''.', method);
    end
    
    %% Evaluate training performance
    % Training predictions
    train_predictions = predict(classifier, features_norm);
    train_accuracy = mean(strcmp(train_predictions, labels));
    
    % Confusion matrix
    [C, order] = confusionmat(labels, train_predictions, 'Order', unique_labels);
    
    if verbose
        fprintf('\nTraining Accuracy: %.1f%%\n', train_accuracy * 100);
        fprintf('\nConfusion Matrix:\n');
        disp_confusion_matrix(C, unique_labels);
    end
    
    %% Cross-validation (optional)
    cv_accuracy = NaN;
    if do_crossval && n_samples >= cv_folds * 2
        if verbose
            fprintf('\nPerforming %d-fold cross-validation...\n', cv_folds);
        end
        
        try
            cv_model = crossval(classifier, 'KFold', cv_folds);
            cv_loss = kfoldLoss(cv_model);
            cv_accuracy = 1 - cv_loss;
            
            if verbose
                fprintf('Cross-validation Accuracy: %.1f%% (+/- estimated)\n', cv_accuracy * 100);
            end
        catch ME
            if verbose
                fprintf('Warning: Cross-validation failed: %s\n', ME.message);
            end
        end
    elseif do_crossval
        if verbose
            fprintf('Skipping cross-validation (insufficient samples)\n');
        end
    end
    
    %% Build output model struct
    model = struct();
    model.classifier = classifier;
    model.method = method;
    model.feature_names = feature_names;
    model.class_names = unique_labels;
    model.n_features = n_features;
    model.n_classes = n_classes;
    model.n_samples = n_samples;
    model.mu = mu;
    model.sigma = sigma;
    model.standardized = do_standardize;
    model.train_accuracy = train_accuracy;
    model.confusion_matrix = C;
    model.cv_accuracy = cv_accuracy;
    model.timestamp = datetime('now');
    
    % Method-specific parameters
    switch lower(method)
        case 'knn'
            model.k = k_neighbors;
        case 'svm'
            model.kernel = svm_kernel;
        case 'tree'
            model.n_leaves = classifier.NumLeaves;
    end
    
    %% Save model to default location
    if isfield(params, 'paths') && isfield(params.paths, 'models')
        model_dir = params.paths.models;
    else
        model_dir = 'models';
    end
    
    if ~exist(model_dir, 'dir')
        mkdir(model_dir);
    end
    
    model_file = fullfile(model_dir, 'gesture_model.mat');
    save(model_file, 'model');
    
    if verbose
        fprintf('\nModel saved to: %s\n', model_file);
        fprintf('===================================\n');
    end
end

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function [features, labels, feature_names] = parse_labeled_data(data_input, verbose)
%PARSE_LABELED_DATA Parse input data from various formats
    
    features = [];
    labels = {};
    feature_names = {};
    
    if ischar(data_input) || isstring(data_input)
        % Input is a file path
        filepath = char(data_input);
        
        if ~exist(filepath, 'file')
            error('ml_train_baseline:FileNotFound', ...
                  'File not found: %s', filepath);
        end
        
        [~, ~, ext] = fileparts(filepath);
        
        if strcmpi(ext, '.mat')
            % Load MAT file
            data = load(filepath);
            
            % Find the data struct (might be nested)
            fields = fieldnames(data);
            if length(fields) == 1
                data = data.(fields{1});
            end
            
            if isfield(data, 'features') && isfield(data, 'labels')
                features = data.features;
                labels = data.labels;
                if isfield(data, 'feature_names')
                    feature_names = data.feature_names;
                end
            elseif isfield(data, 'X') && isfield(data, 'y')
                % Alternative naming convention
                features = data.X;
                labels = data.y;
            else
                error('ml_train_baseline:InvalidFormat', ...
                      'MAT file must contain ''features'' and ''labels'' fields.');
            end
            
        elseif strcmpi(ext, '.csv')
            % Load CSV file
            T = readtable(filepath);
            
            % Find label column
            if ismember('label', T.Properties.VariableNames)
                labels = T.label;
                T.label = [];
            elseif ismember('Label', T.Properties.VariableNames)
                labels = T.Label;
                T.Label = [];
            elseif ismember('gesture', T.Properties.VariableNames)
                labels = T.gesture;
                T.gesture = [];
            else
                error('ml_train_baseline:NoLabelColumn', ...
                      'CSV must have a ''label'' or ''gesture'' column.');
            end
            
            % Convert labels to cell array if needed
            if isnumeric(labels)
                labels = arrayfun(@num2str, labels, 'UniformOutput', false);
            elseif iscategorical(labels)
                labels = cellstr(labels);
            elseif ~iscell(labels)
                labels = cellstr(labels);
            end
            
            % Remaining columns are features
            feature_names = T.Properties.VariableNames;
            features = table2array(T);
            
        else
            error('ml_train_baseline:UnsupportedFormat', ...
                  'Unsupported file format: %s. Use .mat or .csv.', ext);
        end
        
        if verbose
            fprintf('Loaded data from: %s\n', filepath);
        end
        
    elseif isstruct(data_input)
        % Input is a struct
        if isfield(data_input, 'features') && isfield(data_input, 'labels')
            features = data_input.features;
            labels = data_input.labels;
            if isfield(data_input, 'feature_names')
                feature_names = data_input.feature_names;
            elseif isfield(data_input, 'names')
                feature_names = data_input.names;
            end
        else
            error('ml_train_baseline:InvalidStruct', ...
                  'Struct must have ''features'' and ''labels'' fields.');
        end
        
    elseif istable(data_input)
        % Input is a table
        T = data_input;
        
        if ismember('label', T.Properties.VariableNames)
            labels = T.label;
            T.label = [];
        else
            error('ml_train_baseline:NoLabelColumn', ...
                  'Table must have a ''label'' column.');
        end
        
        if iscategorical(labels)
            labels = cellstr(labels);
        end
        
        feature_names = T.Properties.VariableNames;
        features = table2array(T);
        
    else
        error('ml_train_baseline:InvalidInput', ...
              'Input must be a file path, struct, or table.');
    end
    
    % Ensure labels is a column cell array
    if isrow(labels)
        labels = labels';
    end
    
    % Ensure labels are cell strings
    if ~iscell(labels)
        if isnumeric(labels)
            labels = arrayfun(@num2str, labels, 'UniformOutput', false);
        else
            labels = cellstr(labels);
        end
    end
    
    % Generate default feature names if needed
    if isempty(feature_names)
        n_feat = size(features, 2);
        feature_names = arrayfun(@(i) sprintf('feature_%d', i), 1:n_feat, ...
                                'UniformOutput', false);
    end
end

function disp_confusion_matrix(C, class_names)
%DISP_CONFUSION_MATRIX Display confusion matrix with labels
    
    n = length(class_names);
    
    % Truncate long names
    max_len = 10;
    short_names = cellfun(@(s) s(1:min(length(s), max_len)), class_names, ...
                         'UniformOutput', false);
    
    % Header
    fprintf('%12s', 'Actual\\Pred');
    for j = 1:n
        fprintf('%10s', short_names{j});
    end
    fprintf('\n');
    
    % Separator
    fprintf('%s\n', repmat('-', 1, 12 + 10*n));
    
    % Rows
    for i = 1:n
        fprintf('%12s', short_names{i});
        for j = 1:n
            if i == j
                fprintf('%10d', C(i,j));  % Diagonal (correct)
            else
                fprintf('%10d', C(i,j));  % Off-diagonal (errors)
            end
        end
        % Row accuracy
        row_total = sum(C(i,:));
        if row_total > 0
            row_acc = C(i,i) / row_total * 100;
            fprintf('  (%.0f%%)', row_acc);
        end
        fprintf('\n');
    end
end

function val = getFieldOrDefault(s, fieldname, default)
%GETFIELDORDEFAULT Get field value or return default if missing
    if isfield(s, fieldname)
        val = s.(fieldname);
    else
        val = default;
    end
end
