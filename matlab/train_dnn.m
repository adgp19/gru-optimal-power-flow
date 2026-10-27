%% %% PART 2 & 3: DNN (80/20 Split) %% %%
clear; clc; close all;
rng(42);
%% Hyperparameters & architecture
inputSize = 60;      
hiddenSize1 = 64;
hiddenSize2 = 64;
hiddenSize3 = 32;
outputSize = 35;     
learningRate = 0.0001;
epochs = 10000;       
batchSize = 32;
disp('Loading and preparing data...');
load('opf_data.mat');
numSamples = size(X_data, 1);
%% Split data: 80% train, 20% test (SEQUENTIALLY)
nTrain = round(0.80 * numSamples);
nTest = numSamples - nTrain;
trainIdx = 1:nTrain;
testIdx  = (nTrain + 1) : numSamples;
Xtrain = X_data(trainIdx,:);
Ytrain = Y_data(trainIdx,:);
Xtest  = X_data(testIdx,:); % Un-normalized test inputs
Ytest  = Y_data(testIdx,:); % Un-normalized test ground truth
%% Normalize (z-score) using training set stats
muX = mean(Xtrain); sX = std(Xtrain);
muY = mean(Ytrain); sY = std(Ytrain);
sX(sX == 0) = 1; % Handle zero-std columns
sY(sY == 0) = 1;
Xtrain_n = (Xtrain - muX) ./ sX;
Ytrain_n = (Ytrain - muY) ./ sY;
Xtest_n  = (Xtest - muX)  ./ sX; % Normalize test inputs using train stats
%% Weight initialization (He for ReLU layers)
W1 = randn(inputSize, hiddenSize1)   * sqrt(2 / inputSize);
b1 = zeros(1, hiddenSize1);
W2 = randn(hiddenSize1, hiddenSize2) * sqrt(2 / hiddenSize1);
b2 = zeros(1, hiddenSize2);
W3 = randn(hiddenSize2, hiddenSize3) * sqrt(2 / hiddenSize2);
b3 = zeros(1, hiddenSize3);
W4 = randn(hiddenSize3, outputSize)  * sqrt(2 / hiddenSize3);
b4 = zeros(1, outputSize);
%% Activation functions and derivatives
relu = @(x) max(0, x);
relu_deriv = @(x) double(x > 0); 
%% Training setup
numBatches = ceil(nTrain / batchSize);
trainingLoss = zeros(epochs,1);
disp('Starting network training from scratch...');
for epoch = 1:epochs
    perm = randperm(nTrain);
    Xtrain_n_shuffled = Xtrain_n(perm,:);
    Ytrain_n_shuffled = Ytrain_n(perm,:);
    
    epoch_loss = 0;
    
    for b = 1:numBatches
        % create mini-batch
        startIdx = (b-1)*batchSize + 1;
        endIdx = min(b*batchSize, nTrain);
        Xb = Xtrain_n_shuffled(startIdx:endIdx, :);
        Yb = Ytrain_n_shuffled(startIdx:endIdx, :);
        B = size(Xb,1);
        
        % ---------- Forward pass ----------
        z1 = Xb * W1 + repmat(b1, B, 1);   a1 = relu(z1);
        z2 = a1 * W2 + repmat(b2, B, 1);  a2 = relu(z2);
        z3 = a2 * W3 + repmat(b3, B, 1);  a3 = relu(z3);
        z4 = a3 * W4 + repmat(b4, B, 1);  ypred = z4;
        
        % ---------- Loss (MSE) ----------
        err = Yb - ypred;
        loss = mean(err.^2, 'all');
        epoch_loss = epoch_loss + loss * B;
        
        % ---------- Backpropagation ----------
        dL_dypred = -2 * (err) / B;
        
        dL_dW4 = a3' * dL_dypred;
        dL_db4 = sum(dL_dypred, 1);
        dL_da3 = dL_dypred * (W4');
        dL_dz3 = dL_da3 .* relu_deriv(z3);
        
        dL_dW3 = a2' * dL_dz3;
        dL_db3 = sum(dL_dz3, 1);
        dL_da2 = dL_dz3 * (W3');
        dL_dz2 = dL_da2 .* relu_deriv(z2);
        
        dL_dW2 = a1' * dL_dz2;
        dL_db2 = sum(dL_dz2, 1);
        dL_da1 = dL_dz2 * (W2');
        dL_dz1 = dL_da1 .* relu_deriv(z1);
        
        dL_dW1 = Xb' * dL_dz1;
        dL_db1 = sum(dL_dz1, 1);
        
        % ---------- Parameter updates ----------
        W1 = W1 - learningRate * dL_dW1; b1 = b1 - learningRate * dL_db1;
        W2 = W2 - learningRate * dL_dW2; b2 = b2 - learningRate * dL_db2;
        W3 = W3 - learningRate * dL_dW3; b3 = b3 - learningRate * dL_db3;
        W4 = W4 - learningRate * dL_dW4; b4 = b4 - learningRate * dL_db4;
    end
    
    trainingLoss(epoch) = epoch_loss / nTrain;
    
    if mod(epoch, 100) == 0 || epoch==1
        fprintf('Epoch %4d / %4d  TrainMSE = %.6f\n', ...
            epoch, epochs, trainingLoss(epoch));
    end
end
disp('Training complete.');
%% %% --- Part 2: Plot Training Loss --- %% %%
figure('Name', 'Part 2: Training Loss');
plot(1:epochs, trainingLoss, 'LineWidth', 2);
xlabel('Epoch'); ylabel('MSE (normalized units)'); legend('Training Loss');
title('Part 2: Training Loss vs. Epochs');
grid on;
set(gca, 'YScale', 'log');

%% %% --- Part 3: Evaluate on Test Set --- %% %%
B_test = size(Xtest_n, 1);
Z1 = Xtest_n * W1 + repmat(b1, B_test, 1); A1 = relu(Z1);
Z2 = A1 * W2 + repmat(b2, B_test, 1); A2 = relu(Z2);
Z3 = A2 * W3 + repmat(b3, B_test, 1); A3 = relu(Z3);
Z4 = A3 * W4 + repmat(b4, B_test, 1);
Ytest_pred_n = Z4; % normalized predictions
% Denormalize for interpretability
Ytest_pred = Ytest_pred_n .* sY + muY;
Ytest_true = Ytest; % original denormalized ground truth
%% %% --- Part 3: MSE Table --- %% %%
mse_per_output = mean((Ytest_true - Ytest_pred).^2, 1); % [1 x 35]
% MODIFIED: Updated for 35 OPF variables
output_labels = [
    "P_chp"; "U_chp"; "SdnChp"; "SupChp"; "P_ds"; "U_ds"; "SdnDs"; "SupDs";
    "P_ng"; "U_ng"; "SdnNg"; "SupNg"; "Pw1_used"; "Pw2_used"; "Pv1_used"; "Pv2_used";
    "P_b1"; "P_b2"; "E_b1"; "E_b2"; "del1"; "del2"; "del3"; "V1"; "V2"; "V3";
    "Q_chp"; "Q_ds"; "Q_ng"; "Q_w1"; "Q_w2"; "Q_pv1"; "Q_pv2"; "Q_b1"; "Q_b2"
];
mse_table = table(output_labels, mse_per_output');
mse_table.Properties.VariableNames = ["OutputVariable", "Test_MSE"];
disp('Mean Squared Error (MSE) for each output:');
disp(mse_table);

%% %% --- Part 3: Comparison Plots (Entire Test Set) --- %% %%
numTestSamples = size(Xtest, 1);
time_axis_full = 1:numTestSamples;

% --- Figure 1: Generator Active Power ---
figure('Name', 'Part 3: Generator Active Power (Full Set)');
sgtitle('Generator Active Power (Full Test Set)');
subplot(3, 1, 1);
plot(time_axis_full, Ytest_true(:, 1), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_full, Ytest_pred(:, 1), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{1}); % Pchp
ylabel('Power (MW)');
grid on;
subplot(3, 1, 2);
plot(time_axis_full, Ytest_true(:, 5), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_full, Ytest_pred(:, 5), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{5}); % Pds
ylabel('Power (MW)');
grid on;
subplot(3, 1, 3);
plot(time_axis_full, Ytest_true(:, 9), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_full, Ytest_pred(:, 9), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{9}); % Png
xlabel('Test Sample Index');
ylabel('Power (MW)');
grid on;
% --- Figure 2: Photovoltaic (PV) Power ---
figure('Name', 'Part 3: PV Power (Full Set)');
sgtitle('Photovoltaic (PV) Power Dispatch (Full Test Set)');
subplot(2, 1, 1);
plot(time_axis_full, Ytest_true(:, 15), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_full, Ytest_pred(:, 15), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{15}); % Pv1_dispatch
ylabel('Power (MW)');
grid on;
subplot(2, 1, 2);
plot(time_axis_full, Ytest_true(:, 16), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_full, Ytest_pred(:, 16), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{16}); % Pv2_dispatch
xlabel('Test Sample Index');
ylabel('Power (MW)');
grid on;
% --- Figure 3: Battery Power & Energy ---
figure('Name', 'Part 3: Battery Power & Energy (Full Set)');
sgtitle('Battery Power & Energy (Full Test Set)');
subplot(4, 1, 1);
plot(time_axis_full, Ytest_true(:, 17), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_full, Ytest_pred(:, 17), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{17}); % Pb1
ylabel('Power (MW)');
grid on;
subplot(4, 1, 2);
plot(time_axis_full, Ytest_true(:, 18), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_full, Ytest_pred(:, 18), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{18}); % Pb2
ylabel('Power (MW)');
grid on;
subplot(4, 1, 3);
plot(time_axis_full, Ytest_true(:, 19), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_full, Ytest_pred(:, 19), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{19}); % Eb1
ylabel('Energy (MWh)');
grid on;
subplot(4, 1, 4);
plot(time_axis_full, Ytest_true(:, 20), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_full, Ytest_pred(:, 20), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{20}); % Eb2
xlabel('Test Sample Index');
ylabel('Energy (MWh)');
grid on;
% --- Figure 4: Bus Voltages ---
figure('Name', 'Part 3: Bus Voltages (Full Set)');
sgtitle('Bus Voltages (Full Test Set)');
subplot(3, 1, 1);
plot(time_axis_full, Ytest_true(:, 24), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_full, Ytest_pred(:, 24), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{24}); % V1
ylabel('Voltage (kV)');
grid on;
subplot(3, 1, 2);
plot(time_axis_full, Ytest_true(:, 25), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_full, Ytest_pred(:, 25), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{25}); % V2
ylabel('Voltage (kV)');
grid on;
subplot(3, 1, 3);
plot(time_axis_full, Ytest_true(:, 26), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_full, Ytest_pred(:, 26), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{26}); % V3
xlabel('Test Sample Index');
ylabel('Voltage (kV)');
grid on;
% --- Figure 5: Reactive Powers ---
figure('Name', 'Part 3: Reactive Powers (Full Set)');
sgtitle('Reactive Powers (Q) (Full Test Set)');
q_indices = [27, 28, 29, 30, 31, 32, 33, 34, 35];
for k = 1:9
    idx = q_indices(k);
    subplot(3, 3, k);
    plot(time_axis_full, Ytest_true(:, idx), 'b-'); hold on;
    plot(time_axis_full, Ytest_pred(:, idx), 'r--');
    title(output_labels{idx});
    grid on;
    if k == 1
        legend('Truth', 'Pred', 'Location', 'best');
    end
    if k > 6
        xlabel('Test Sample Index');
    end
    if mod(k, 3) == 1
        ylabel('Q (MVar)');
    end
end
%% %% --- Part 3: 40% Test Set Plots --- %% %%
num_40_percent = round(0.4 * numTestSamples);
if num_40_percent < 2
    warning('Test set is too small to take 40%. Skipping this plot.');
    return;
end
time_axis_40 = 1:num_40_percent;
Ytest_true_40 = Ytest_true(1:num_40_percent, :);
Ytest_pred_40 = Ytest_pred(1:num_40_percent, :);

% --- Figure 6: Generator Active Power (40% Set) ---
figure('Name', 'Part 3: Generator Active Power (40% Set)');
sgtitle('Generator Active Power (First 40% of Test Samples)');
subplot(3, 1, 1);
plot(time_axis_40, Ytest_true_40(:, 1), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_40, Ytest_pred_40(:, 1), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{1}); % Pchp
ylabel('Power (MW)');
grid on;
subplot(3, 1, 2);
plot(time_axis_40, Ytest_true_40(:, 5), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_40, Ytest_pred_40(:, 5), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{5}); % Pds
ylabel('Power (MW)');
grid on;
subplot(3, 1, 3);
plot(time_axis_40, Ytest_true_40(:, 9), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_40, Ytest_pred_40(:, 9), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{9}); % Png
xlabel('Test Sample Index');
ylabel('Power (MW)');
grid on;
% --- Figure 7: Photovoltaic (PV) Power (40% Set) ---
figure('Name', 'Part 3: PV Power (40% Set)');
sgtitle('Photovoltaic (PV) Power Dispatch (First 40% of Test Samples)');
subplot(2, 1, 1);
plot(time_axis_40, Ytest_true_40(:, 15), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_40, Ytest_pred_40(:, 15), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{15}); % Pv1_dispatch
ylabel('Power (MW)');
grid on;
subplot(2, 1, 2);
plot(time_axis_40, Ytest_true_40(:, 16), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_40, Ytest_pred_40(:, 16), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{16}); % Pv2_dispatch
xlabel('Test Sample Index');
ylabel('Power (MW)');
grid on;
% --- Figure 8: Battery Power & Energy (40% Set) ---
figure('Name', 'Part 3: Battery Power & Energy (40% Set)');
sgtitle('Battery Power & Energy (First 40% of Test Samples)');
subplot(4, 1, 1);
plot(time_axis_40, Ytest_true_40(:, 17), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_40, Ytest_pred_40(:, 17), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{17}); % Pb1
ylabel('Power (MW)');
grid on;
subplot(4, 1, 2);
plot(time_axis_40, Ytest_true_40(:, 18), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_40, Ytest_pred_40(:, 18), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{18}); % Pb2
ylabel('Power (MW)');
grid on;
subplot(4, 1, 3);
plot(time_axis_40, Ytest_true_40(:, 19), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_40, Ytest_pred_40(:, 19), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{19}); % Eb1
ylabel('Energy (MWh)');
grid on;
subplot(4, 1, 4);
plot(time_axis_40, Ytest_true_40(:, 20), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_40, Ytest_pred_40(:, 20), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{20}); % Eb2
xlabel('Test Sample Index');
ylabel('Energy (MWh)');
grid on;
% --- Figure 9: Bus Voltages (40% Set) ---
figure('Name', 'Part 3: Bus Voltages (40% Set)');
sgtitle('Bus Voltages (First 40% of Test Samples)');
subplot(3, 1, 1);
plot(time_axis_40, Ytest_true_40(:, 24), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_40, Ytest_pred_40(:, 24), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{24}); % V1
ylabel('Voltage (kV)');
grid on;
subplot(3, 1, 2);
plot(time_axis_40, Ytest_true_40(:, 25), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_40, Ytest_pred_40(:, 25), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{25}); % V2
ylabel('Voltage (kV)');
grid on;
subplot(3, 1, 3);
plot(time_axis_40, Ytest_true_40(:, 26), 'b-', 'LineWidth', 1.5); hold on;
plot(time_axis_40, Ytest_pred_40(:, 26), 'r--', 'LineWidth', 1.5);
legend('Ground Truth', 'Prediction', 'Location', 'best');
title(output_labels{26}); % V3
xlabel('Test Sample Index');
ylabel('Voltage (kV)');
grid on;
% --- Figure 10: Reactive Powers (40% Set) ---
figure('Name', 'Part 3: Reactive Powers (40% Set)');
sgtitle('Reactive Powers (Q) (First 40% of Test Samples)');
for k = 1:9
    idx = q_indices(k);
    subplot(3, 3, k);
    plot(time_axis_40, Ytest_true_40(:, idx), 'b-'); hold on;
    plot(time_axis_40, Ytest_pred_40(:, idx), 'r--');
    title(output_labels{idx});
    grid on;
    if k == 1
        legend('Truth', 'Pred', 'Location', 'best');
    end
    if k > 6
        xlabel('Test Sample Index');
    end
    if mod(k, 3) == 1
        ylabel('Q (MVar)');
    end
end
fprintf('Part 3 plotting complete. 10 figures created.\n');
