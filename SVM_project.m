close all;
%clear all;
clc;

% SVM FOR MACHINE LEARNING PROJECT

% % Image directory.
% image_dir_norm = ['C:\Users\Samuel Washburn\Documents\JHU Masters\Spring 2023\' ...
%     'Mach Learning\ML PROJECT\data_folders\raw_data_kaggle\train\NORMAL'];
% image_dir_pneu = ['C:\Users\Samuel Washburn\Documents\JHU Masters\Spring 2023\' ...
%     'Mach Learning\ML PROJECT\data_folders\raw_data_kaggle\train\PNEUMONIA'];
% 
% fileformat_norm = fullfile(image_dir_norm, '*.jpeg');
% fileformat_pneu = fullfile(image_dir_pneu, '*.jpeg');
% 
% filenames_norm = dir(fileformat_norm);
% filenames_pneu = dir(fileformat_pneu);
% 
% num_norm = length(filenames_norm);
% num_pneu = length(filenames_pneu);
% 
% image_size = 256;
% norm_images = zeros(num_norm, image_size^2);
% pneu_images = zeros(num_pneu, image_size^2);
% norm_labels = ones(num_norm, 1);
% pneu_labels = ones(num_pneu, 1) * 2;
% 
% for k = 1:num_norm
%     im_name = filenames_norm(k).name;
%     im_path = fullfile(image_dir_norm, im_name);
%     im_norm = imread(im_path);
%     im_norm = imresize(im_norm, [image_size image_size]);
%     im_norm = reshape(im_norm, [1 image_size^2]);
%     norm_images(k,:) = im_norm;
% end
% norm_images = norm_images / 255;
% 
% for k = 1:num_pneu
%     im_name = filenames_pneu(k).name;
%     im_path = fullfile(image_dir_pneu, im_name);
%     im_pneu = imread(im_path);
%     im_pneu = imresize(im_pneu, [image_size image_size]);
%     im_pneu = reshape(im_pneu, [1 image_size^2]);
%     pneu_images(k,:) = im_pneu;
% end
% pneu_images = pneu_images / 255;
% 

% test_percent = 0.1;
% test_count_norm = round(test_percent * num_norm);
% test_count_pneu = round(test_percent * num_pneu);
% 
% train_count_norm = num_norm - test_count_norm;
% train_count_pneu = num_pneu - test_count_pneu;
% 
% train_images = zeros(train_count_norm+train_count_pneu, image_size^2);
% train_labels = zeros(train_count_norm+train_count_pneu, 1);
% 
% test_images = zeros(test_count_norm+test_count_pneu, image_size^2);
% test_labels = zeros(test_count_norm+test_count_pneu, 1);
% 
% test_images(1:test_count_norm, :) = norm_images(1:test_count_norm, :);
% test_images(test_count_norm+1:end, :) = ...
%     pneu_images(1:test_count_pneu, :);
% test_labels(1:test_count_norm) = norm_labels(1:test_count_norm);
% test_labels(test_count_norm+1:end) = pneu_labels(1:test_count_pneu);
% 
% train_images(1:train_count_norm, :) = norm_images(test_count_norm+1:end, :);
% train_images(train_count_norm+1:end, :) = ...
%     pneu_images(test_count_pneu+1:end, :);
% train_labels(1:train_count_norm) = norm_labels(test_count_norm+1:end);
% train_labels(train_count_norm+1:end) = pneu_labels(test_count_pneu+1:end);


% input_images = zeros(num_norm+num_pneu, image_size^2);
% input_images(1:num_norm, :) = norm_images;
% input_images((num_norm+1):(num_norm+num_pneu), :) = pneu_images;
% input_labels = zeros(num_norm+num_pneu, 1);
% input_labels(1:num_norm, :) = norm_labels;
% input_labels((num_norm+1):(num_norm+num_pneu), :) = pneu_labels;

% [~,idx] = sort(train_images(:,30000));
% train_images = train_images(idx,:);
% train_labels = train_labels(idx,:);

% test_images = input_images(1:100,:);
% test_labels = input_labels(1:100,:);
% train_images = input_images(101:end,:);
% train_labels = input_labels(101:end,:);

% Use the built in MATLAB function to run PCA on training data.
% [coeff_train, score_train, ~, ~, explained, mu] = pca(train_images);

% score_test = (test_images - mu) * coeff_train;

L = [10 30 50 70 90];
B = [0.009 0.09 0.9];

% Store the index of the component that is associated with each total
% variance threshold.
IDX = zeros(1,length(L));

% Percent of total variance.
ACC = zeros(1,length(L));

perform_svm_linear = zeros(length(B),length(L));
perform_svm_gaus = zeros(length(B),length(L));
perform_svm_poly = zeros(length(B),length(L));

run_svm = 1;

%% Implementation of built in SVM algorithm.

if run_svm

    for B_ind = 1:length(B)

        for i = 1:length(L)
    
            disp("Starting built in SVM algorithm for Total Variance = " + L(i))
        
            % Find the number of principal components to reach the current
            % total variance threshold.
            index = find(cumsum(explained)>L(i),1);
        
            % Save the number of components and the percent variance.
            IDX(i) = index;
            ACC(i) = L(i);
        
            % Save the first L components of the projected training data.
            score_train_L = score_train(:,1:index);
        
            % Save the first L components of the projected test data.
            score_test_L = score_test(:,1:index);
    
            % Create the template for the SVM.
            t_linear = templateSVM('Standardize',true,'KernelFunction','linear', 'BoxConstraint', B(B_ind));
            t_gaus = templateSVM('Standardize',true,'KernelFunction','gaussian', 'BoxConstraint', B(B_ind));
            t_poly = templateSVM('Standardize',true,'KernelFunction','polynomial', 'BoxConstraint', B(B_ind));
    
            % Train the classification model using the template and
            % training data.
            Mdl_svm_linear = fitcecoc(score_train_L, train_labels,'Learners',t_linear);
            Mdl_svm_gaus = fitcecoc(score_train_L, train_labels,'Learners',t_gaus);
            Mdl_svm_poly = fitcecoc(score_train_L, train_labels,'Learners',t_poly);
    
            % Classify the test data using the trained SVM model.
            [pred_labels_linear,score_result_linear,cost_linear] = predict(Mdl_svm_linear,score_test_L);
            perform_svm_linear(B_ind,i) = sum(pred_labels_linear == test_labels)/size(test_labels,1);
            [pred_labels_gaus,score_result_gaus,cost_gaus] = predict(Mdl_svm_gaus,score_test_L);
            perform_svm_gaus(B_ind,i) = sum(pred_labels_gaus == test_labels)/size(test_labels,1);
            [pred_labels_poly,score_result_poly,cost_poly] = predict(Mdl_svm_poly,score_test_L);
            perform_svm_poly(B_ind,i) = sum(pred_labels_poly == test_labels)/size(test_labels,1);
    
            disp("Finished built in SVM algorithm for Total Variance = " + L(i))
        end
    
        % Setup performance values which will be loaded into table for display.
        Percent_of_Variance = ACC';
        Performance_linear = transpose(perform_svm_linear(B_ind,:));
        Performance_gaus = transpose(perform_svm_gaus(B_ind,:));
        Performance_poly = transpose(perform_svm_poly(B_ind,:));
        NumBases = IDX';
    
        % Display performance of custom knn for different total PCA variances.
        svm_tab_linear = table(Percent_of_Variance, Performance_linear, NumBases)
        svm_tab_gaus = table(Percent_of_Variance, Performance_gaus, NumBases)
        svm_tab_poly = table(Percent_of_Variance, Performance_poly, NumBases)
    end
end

test = 0;
for i = 1:length(test_labels)
    if test_labels(i) == 1
        test = test+1;
    end
end
