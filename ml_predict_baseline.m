%% ML_PREDICT_BASELINE - Predict Gesture Using Trained ML Model
% Loads a trained model and predicts the gesture label for extracted features.
%
% SYNTAX:
%   cls = ml_predict_baseline(feat, params)
%
% INPUTS:
%   feat   - Feature struct from extract_features()
%            .x     : 1xM feature vector
%            .names : 1xM cell array of feature names
%   params - Configuration struct from config_params()
%
% OUTPUTS:
%   cls    - Classification result struct
%            .label  : Predicted gesture name (string)
%            .score  : Confidence score (0-1)
%            .method : "ml"
%            .reason : Explanation of prediction
%            .probs  : Per-class probabilities (if available)
%
% NOTES:
%   - Looks for model file in models/gesture_model_latest.mat
%   - Falls back to rule-based classifier if no model found
%   - Applies same normalization used during training
%
% See also: ml_train_baseline, classify_gesture_rules, extract_features

function cls = ml_predict_baseline(feat, params)
    %% ========================================================================
    %  INITIALIZATION
    %  ========================================================================
    
    cls = struct();
    cls.label = 'unknown';
    cls.score = 0;
    cls.method = 'ml';
    cls.reason = '';
    cls.probs = [];
    
    %% ========================================================================
    %  LOAD MODEL
    %  ========================================================================
    
    % Find model file
    thisFile = mfilename('fullpath');
    [thisDir, ~, ~] = fileparts(thisFile);
    srcDir = fileparts(thisDir);
    repoDir = fileparts(srcDir);
    
    modelFile = fullfile(repoDir, 'models', 'gesture_model_latest.mat');
    
    % Check for custom model path in params
    if isfield(params, 'ml') && isfield(params.ml, 'modelFile') && ~isempty(params.ml.modelFile)
        if exist(params.ml.modelFile, 'file')
            modelFile = params.ml.modelFile;
        end
    end
    
    if ~exist(modelFile, 'file')
        warning('ML model not found: %s\nFalling back to rule-based classifier.', modelFile);
        cls = classify_gesture_rules(feat, params);
        cls.reason = [cls.reason, ' (ML model not found, used rules)'];
        return;
    end
    
    % Load model
    try
        loaded = load(modelFile);
    catch ME
        warning('Failed to load ML model: %s\nFalling back to rule-based classifier.', ME.message);
        cls = classify_gesture_rules(feat, params);
        cls.reason = [cls.reason, ' (ML model load failed, used rules)'];
        return;
    end
    
    if ~isfield(loaded, 'model')
        warning('Invalid model file format.\nFalling back to rule-based classifier.');
        cls = classify_gesture_rules(feat, params);
        cls.reason = [cls.reason, ' (invalid model format, used rules)'];
        return;
    end
    
    model = loaded.model;
    
    %% ========================================================================
    %  PREPARE FEATURES
    %  ========================================================================
    
    X = feat.x;
    
    % Check feature dimensions
    if isfield(loaded, 'featureNames')
        expectedFeatures = length(loaded.featureNames);
        if length(X) ~= expectedFeatures
            warning('Feature dimension mismatch: expected %d, got %d.\nFalling back to rule-based classifier.', ...
                    expectedFeatures, length(X));
            cls = classify_gesture_rules(feat, params);
            cls.reason = [cls.reason, ' (feature mismatch, used rules)'];
            return;
        end
    end
    
    % Apply normalization (same as training)
    if isfield(loaded, 'featureMean') && isfield(loaded, 'featureStd')
        X_norm = (X - loaded.featureMean) ./ loaded.featureStd;
    else
        X_norm = X;
    end
    
    % Handle NaN/Inf
    X_norm(~isfinite(X_norm)) = 0;
    
    %% ========================================================================
    %  PREDICT
    %  ========================================================================
    
    try
        % Check model type and predict accordingly
        if isstruct(model) && isfield(model, 'type')
            % Custom struct-based model (from ml_train_baseline)
            switch model.type
                case 'knn'
                    % Manual kNN prediction
                    distances = sqrt(sum((model.X - X_norm).^2, 2));
                    [~, sortIdx] = sort(distances);
                    kNearest = sortIdx(1:min(model.k, length(sortIdx)));
                    nearestLabels = model.Y(kNearest);
                    
                    % Majority vote
                    uniqueLabels = categories(nearestLabels);
                    votes = zeros(length(uniqueLabels), 1);
                    for i = 1:length(uniqueLabels)
                        votes(i) = sum(nearestLabels == uniqueLabels{i});
                    end
                    [maxVotes, maxIdx] = max(votes);
                    
                    cls.label = char(uniqueLabels{maxIdx});
                    cls.score = maxVotes / length(kNearest);
                    cls.probs = votes / sum(votes);
                    cls.reason = sprintf('kNN (k=%d): %d/%d neighbors voted %s', ...
                                        model.k, maxVotes, length(kNearest), cls.label);
                    
                case 'tree'
                    % Decision tree (MATLAB fitctree model)
                    [label, scores] = predict(model.classifier, X_norm);
                    cls.label = char(label);
                    cls.score = max(scores);
                    cls.probs = scores;
                    cls.reason = sprintf('Decision tree: %.1f%% confidence', cls.score * 100);
                    
                case 'svm'
                    % SVM (MATLAB fitcsvm or fitcecoc model)
                    [label, scores] = predict(model.classifier, X_norm);
                    cls.label = char(label);
                    if size(scores, 2) > 1
                        % Multi-class: scores are per-class
                        cls.score = max(scores);
                        cls.probs = scores;
                    else
                        % Binary: convert decision value to pseudo-probability
                        cls.score = 1 / (1 + exp(-scores));
                    end
                    cls.reason = sprintf('SVM: %.1f%% confidence', cls.score * 100);
                    
                otherwise
                    error('Unknown model type: %s', model.type);
            end
            
        elseif isa(model, 'ClassificationKNN')
            % MATLAB's built-in kNN classifier
            [label, scores, cost] = predict(model, X_norm);
            cls.label = char(label);
            cls.score = max(scores);
            cls.probs = scores;
            cls.reason = sprintf('kNN classifier: %.1f%% posterior', cls.score * 100);
            
        elseif isa(model, 'ClassificationTree')
            % MATLAB's built-in decision tree
            [label, scores] = predict(model, X_norm);
            cls.label = char(label);
            cls.score = max(scores);
            cls.probs = scores;
            cls.reason = sprintf('Decision tree: %.1f%% confidence', cls.score * 100);
            
        elseif isa(model, 'ClassificationSVM') || isa(model, 'ClassificationECOC')
            % MATLAB's built-in SVM
            [label, scores] = predict(model, X_norm);
            cls.label = char(label);
            cls.score = max(scores);
            cls.probs = scores;
            cls.reason = sprintf('SVM: %.1f%% confidence', cls.score * 100);
            
        elseif isa(model, 'ClassificationEnsemble')
            % Ensemble classifier (e.g., Random Forest)
            [label, scores] = predict(model, X_norm);
            cls.label = char(label);
            cls.score = max(scores);
            cls.probs = scores;
            cls.reason = sprintf('Ensemble: %.1f%% confidence', cls.score * 100);
            
        else
            % Try generic predict
            try
                [label, scores] = predict(model, X_norm);
                cls.label = char(label);
                cls.score = max(scores);
                cls.probs = scores;
                cls.reason = sprintf('ML classifier: %.1f%% confidence', cls.score * 100);
            catch
                error('Unsupported model type: %s', class(model));
            end
        end
        
    catch ME
        warning('ML prediction failed: %s\nFalling back to rule-based classifier.', ME.message);
        cls = classify_gesture_rules(feat, params);
        cls.reason = [cls.reason, sprintf(' (ML prediction failed: %s)', ME.message)];
        return;
    end
    
    %% ========================================================================
    %  POST-PROCESSING
    %  ========================================================================
    
    % Confidence threshold check
    if isfield(params, 'ml') && isfield(params.ml, 'minConfidence')
        if cls.score < params.ml.minConfidence
            originalLabel = cls.label;
            originalScore = cls.score;
            
            % Fall back to rules for low-confidence predictions
            clsRules = classify_gesture_rules(feat, params);
            
            if clsRules.score > cls.score
                cls = clsRules;
                cls.reason = sprintf('ML confidence too low (%.1f%% for %s), used rules instead', ...
                                    originalScore * 100, originalLabel);
            else
                cls.reason = [cls.reason, sprintf(' (low confidence, but better than rules)')];
            end
        end
    end
    
    % Validate label against known gestures
    if isfield(params, 'gestures') && isfield(params.gestures, 'labels')
        knownLabels = params.gestures.labels;
        if ~any(strcmpi(cls.label, knownLabels)) && ~strcmpi(cls.label, 'unknown')
            cls.reason = [cls.reason, ' (warning: label not in known gesture set)'];
        end
    end
end
