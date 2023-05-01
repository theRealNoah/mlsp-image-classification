close all;
%clear all;
clc;

% Machine Learning for Signal Processing
% Course Project: Classifying Chest X-rays for Pneumonia
% Noah Hamilton and Samuel Washburn

% % Specify the directories where the images are located.
% dir_train_norm = ['C:\Users\Samuel Washburn\Documents\JHU Masters\Spring 2023\' ...
%     'Mach Learning\ML PROJECT\data_folders\train\NORMAL'];
% dir_train_pneu = ['C:\Users\Samuel Washburn\Documents\JHU Masters\Spring 2023\' ...
%     'Mach Learning\ML PROJECT\data_folders\train\PNEUMONIA'];
% dir_test_norm = ['C:\Users\Samuel Washburn\Documents\JHU Masters\Spring 2023\' ...
%     'Mach Learning\ML PROJECT\data_folders\test\NORMAL'];
% dir_test_pneu = ['C:\Users\Samuel Washburn\Documents\JHU Masters\Spring 2023\' ...
%     'Mach Learning\ML PROJECT\data_folders\test\PNEUMONIA'];
% 
% % Full filenames for each image.
% full_dir_train_norm = fullfile(dir_train_norm, '*.jpeg');
% full_dir_train_pneu = fullfile(dir_train_pneu, '*.jpeg');
% full_dir_test_norm = fullfile(dir_test_norm, '*.jpeg');
% full_dir_test_pneu = fullfile(dir_test_pneu, '*.jpeg');
% 
% % Create structure for each image.
% files_train_norm = dir(full_dir_train_norm);
% files_train_pneu = dir(full_dir_train_pneu);
% files_test_norm = dir(full_dir_test_norm);
% files_test_pneu = dir(full_dir_test_pneu);
% 
% % Number of images in each directory.
% num_train_norm = length(files_train_norm);
% num_train_pneu = length(files_train_pneu);
% num_test_norm = length(files_test_norm);
% num_test_pneu = length(files_test_pneu);
% 
% % Create matrices to store all of the images.
% % Image size determines how the images will be shrunk and reshaped.
% image_size = 256;
% train_norm_images = zeros(num_train_norm, image_size^2);
% train_pneu_images = zeros(num_train_pneu, image_size^2);
% test_norm_images = zeros(num_test_norm, image_size^2);
% test_pneu_images = zeros(num_test_pneu, image_size^2);
% 
% % Create labels for the images.
% % Normal images will be labeled 1.
% % Pneumonia images will be labeled 2.
% train_norm_labels = ones(num_train_norm, 1);
% train_pneu_labels = ones(num_train_pneu, 1) * 2;
% test_norm_labels = ones(num_test_norm, 1);
% test_pneu_labels = ones(num_test_pneu, 1) * 2;
% 
% % Reshape the images so they are all the same size.
% % Normalize the images.
% 
% % Trim edges of images that are not the chest cavity, because they are
% % not relevant to detecting pneumonia.
% trim = 0;
% top_trim = 0.15;
% bottom_trim = 0.85;
% left_trim = 0.15;
% right_trim = 0.85;
% 
% % Perform histogram equalization on the images.
% histogram_equal = 0;
% 
% for i = 1:num_train_norm
%     im_name = files_train_norm(i).name;
%     im_path = fullfile(dir_train_norm, im_name);
%     im_load = imread(im_path);
%     if trim == 1
%         [im_M, im_N] = size(im_load);
%         top_bound = round(top_trim * im_M);
%         bottom_bound = round(bottom_trim * im_M);
%         left_bound = round(left_trim * im_N);
%         right_bound = round(right_trim * im_N);
%         im_load = im_load(top_bound:bottom_bound,left_bound:right_bound);
%     end
%     if histogram_equal == 1
%         im_load = histeq(im_load);
%     end
%     im_load = imresize(im_load, [image_size image_size]);
%     im_load = reshape(im_load, [1 image_size^2]);
%     train_norm_images(i,:) = im_load;
% end
% train_norm_images = train_norm_images / 255;
% 
% for i = 1:num_train_pneu
%     im_name = files_train_pneu(i).name;
%     im_path = fullfile(dir_train_pneu, im_name);
%     im_load = imread(im_path);
%     if trim == 1
%         [im_M, im_N] = size(im_load);
%         top_bound = round(top_trim * im_M);
%         bottom_bound = round(bottom_trim * im_M);
%         left_bound = round(left_trim * im_N);
%         right_bound = round(right_trim * im_N);
%         im_load = im_load(top_bound:bottom_bound,left_bound:right_bound);
%     end
%     if histogram_equal == 1
%         im_load = histeq(im_load);
%     end
%     im_load = imresize(im_load, [image_size image_size]);
%     im_load = reshape(im_load, [1 image_size^2]);
%     train_pneu_images(i,:) = im_load;
% end
% train_pneu_images = train_pneu_images / 255;
% 
% for i = 1:num_test_norm
%     im_name = files_test_norm(i).name;
%     im_path = fullfile(dir_test_norm, im_name);
%     im_load = imread(im_path);
%     if trim == 1
%         [im_M, im_N] = size(im_load);
%         top_bound = round(top_trim * im_M);
%         bottom_bound = round(bottom_trim * im_M);
%         left_bound = round(left_trim * im_N);
%         right_bound = round(right_trim * im_N);
%         im_load = im_load(top_bound:bottom_bound,left_bound:right_bound);
%     end
%     if histogram_equal == 1
%         im_load = histeq(im_load);
%     end
%     im_load = imresize(im_load, [image_size image_size]);
%     im_load = reshape(im_load, [1 image_size^2]);
%     test_norm_images(i,:) = im_load;
% end
% test_norm_images = test_norm_images / 255;
% 
% for i = 1:num_test_pneu
%     im_name = files_test_pneu(i).name;
%     im_path = fullfile(dir_test_pneu, im_name);
%     im_load = imread(im_path);
%     if trim == 1
%         [im_M, im_N] = size(im_load);
%         top_bound = round(top_trim * im_M);
%         bottom_bound = round(bottom_trim * im_M);
%         left_bound = round(left_trim * im_N);
%         right_bound = round(right_trim * im_N);
%         im_load = im_load(top_bound:bottom_bound,left_bound:right_bound);
%     end
%     if histogram_equal == 1
%         im_load = histeq(im_load);
%     end
%     im_load = imresize(im_load, [image_size image_size]);
%     im_load = reshape(im_load, [1 image_size^2]);
%     test_pneu_images(i,:) = im_load;
% end
% test_pneu_images = test_pneu_images / 255;
% 
% disp("Finished loading and formatting images.")
% 
% % Combine the normal and pneumonia images into a single training matrix
% % and single test matrix.
% 
% train_images = zeros(num_train_norm + num_train_pneu, image_size^2);
% train_labels = zeros(num_train_norm + num_train_pneu, 1);
% test_images = zeros(num_test_norm + num_test_pneu, image_size^2);
% test_labels = zeros(num_test_norm + num_test_pneu, 1);
% 
% train_images(1:num_train_norm,:) = train_norm_images;
% train_images(num_train_norm+1:end,:) = train_pneu_images;
% train_labels(1:num_train_norm,1) = train_norm_labels;
% train_labels(num_train_norm+1:end,:) = train_pneu_labels;
% 
% test_images(1:num_test_norm,:) = test_norm_images;
% test_images(num_test_norm+1:end,:) = test_pneu_images;
% test_labels(1:num_test_norm,1) = test_norm_labels;
% test_labels(num_test_norm+1:end,:) = test_pneu_labels;
% 
% 
% % Use the built in MATLAB function to run PCA on training data.
% [coeff_train, score_train, ~, ~, explained, mu] = pca(train_images);
% 
% disp("Finished running PCA.")
% 
% % Compute the project test images using the directions from PCA.
% score_test = (test_images - mu) * coeff_train;
% 
% % Vary the number of bases we use from PCA based on percent of total
% % variance.
% % L = [30 35 40 45 50 55 60 65 70];
% L = [45];
% 
% % Vary the box constraint value we use for the MATLAB SVM function.
% % B = [0.009 0.09 0.9];
% B = [0.9];
% 
% % Store the index of the component that is associated with each total
% % variance threshold.
% IDX = zeros(1,length(L));
% 
% % Percent of total variance.
% ACC = zeros(1,length(L));
% 
% % Store the performance of each kind of SVM kernel.
% perform_svm_linear = zeros(length(B),length(L));
% perform_svm_gaus = zeros(length(B),length(L));
% perform_svm_poly = zeros(length(B),length(L));
% 
% % Implementation of built in SVM algorithm.
% 
% 
% for B_ind = 1:length(B)
% 
%     for i = 1:length(L)
% 
%         % Find the number of principal components to reach the current
%         % total variance threshold.
%         index = find(cumsum(explained)>L(i),1);
% 
%         % Save the number of components and the percent variance.
%         IDX(i) = index;
%         ACC(i) = L(i);
% 
%         % Save the first L components of the projected training data.
%         score_train_L = score_train(:,1:index);
%         
%         % Save the first L components of the projected test data.
%         score_test_L = score_test(:,1:index);
% 
%         % Create the template for the SVM.
%         t_linear = templateSVM('Standardize',true,'KernelFunction','linear', 'BoxConstraint', B(B_ind));
%         t_gaus = templateSVM('Standardize',true,'KernelFunction','gaussian', 'BoxConstraint', B(B_ind));
%         t_poly = templateSVM('Standardize',true,'KernelFunction','polynomial', 'BoxConstraint', B(B_ind));
% 
%         % Train the classification model using the template and
%         % training data.
%         Mdl_svm_linear = fitcecoc(score_train_L, train_labels,'Learners',t_linear);
%         Mdl_svm_gaus = fitcecoc(score_train_L, train_labels,'Learners',t_gaus);
%         Mdl_svm_poly = fitcecoc(score_train_L, train_labels,'Learners',t_poly);
% 
%         % Classify the test data using the trained SVM model.
%         [pred_labels_linear,score_result_linear,cost_linear] = predict(Mdl_svm_linear,score_test_L);
%         perform_svm_linear(B_ind,i) = sum(pred_labels_linear == test_labels)/size(test_labels,1);
%         [pred_labels_gaus,score_result_gaus,cost_gaus] = predict(Mdl_svm_gaus,score_test_L);
%         perform_svm_gaus(B_ind,i) = sum(pred_labels_gaus == test_labels)/size(test_labels,1);
%         [pred_labels_poly,score_result_poly,cost_poly] = predict(Mdl_svm_poly,score_test_L);
%         perform_svm_poly(B_ind,i) = sum(pred_labels_poly == test_labels)/size(test_labels,1);
% 
%         disp("Finished built in SVM algorithm for Total Variance = " + L(i))
%     end
% 
% %     save linear_svm_3.mat perform_svm_linear
% %     save gaus_svm_3.mat perform_svm_gaus
% %     save poly_svm_3.mat perform_svm_poly
% 
%     % Setup performance values which will be loaded into table for display.
%     Percent_of_Variance = ACC';
%     Performance_linear = transpose(perform_svm_linear(B_ind,:));
%     Performance_gaus = transpose(perform_svm_gaus(B_ind,:));
%     Performance_poly = transpose(perform_svm_poly(B_ind,:));
%     NumBases = IDX';
% 
%     % Display performance of custom knn for different total PCA variances.
%     svm_tab_linear = table(Percent_of_Variance, Performance_linear, NumBases)
%     svm_tab_gaus = table(Percent_of_Variance, Performance_gaus, NumBases)
%     svm_tab_poly = table(Percent_of_Variance, Performance_poly, NumBases)
% 
% end

% Compute confusion matrix values.
act_norm_pred_norm = 0;
act_norm_pred_pneu = 0;
act_pneu_pred_norm = 0;
act_pneu_pred_pneu = 0;
for j = 1:length(test_labels)
    if test_labels(j) == 1
        if pred_labels_linear(j) == 1
            act_norm_pred_norm = act_norm_pred_norm + 1;
        else
            act_norm_pred_pneu = act_norm_pred_pneu + 1;
        end
    end
    if test_labels(j) == 2
        if pred_labels_linear(j) == 2
            act_pneu_pred_pneu = act_pneu_pred_pneu + 1;
        else
            act_pneu_pred_norm = act_pneu_pred_norm + 1;
        end
    end
end

% load linear_svm_1 perform_svm_linear
% load gaus_svm_1 perform_svm_gaus
% load poly_svm_1 perform_svm_poly

% figure
% plot(L, perform_svm_linear, 'LineWidth', 2.5)
% lgd = legend(num2str(B(1)), num2str(B(2)), num2str(B(3)))
% lgd.Title.String = 'Box Constraint';
% fontsize(gca,20,"pixels")
% grid on
% 
% figure
% plot(L, perform_svm_gaus, 'LineWidth', 2.5)
% lgd = legend(num2str(B(1)), num2str(B(2)), num2str(B(3)))
% lgd.Title.String = 'Box Constraint';
% fontsize(gca,20,"pixels")
% grid on
% 
% figure
% plot(L, perform_svm_poly, 'LineWidth', 2.5)
% lgd = legend(num2str(B(1)), num2str(B(2)), num2str(B(3)))
% lgd.Title.String = 'Box Constraint';
% fontsize(gca,20,"pixels")
% grid on