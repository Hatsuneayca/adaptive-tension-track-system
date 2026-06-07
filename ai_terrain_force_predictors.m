% 1. Bir kere çalıştır
fis = create_fis_terrain_force();
net = train_nn_terrain_force();

% 2. Predictor fonksiyonlarını hazırla
pred_fuzzy = @(v, sv, e) fuzzy_predict_ft(fis, v, sv, e);
pred_nn    = @(v, sv, e) nn_predict_ft(net, v, sv, e);

% 3. Simülasyonda kullan
F_pred = pred_fuzzy(vehicle_velocity(i), susp_var(i), curr_err);
% veya
F_pred = pred_nn(vehicle_velocity(i), susp_var(i), curr_err);